"""Drive EFabric's web UI with Playwright and the installed Chrome, in its own profile (runs/pm/efabric-profile).

    runs/pm/.venv/bin/python runs/pm/bin/efabric.py login      # opens Chrome; sign in, then close the window
    runs/pm/.venv/bin/python runs/pm/bin/efabric.py inspect [URL]  # read-only: prints the page's buttons, fields and rows
Only efabric.cs.wisc.edu is ever opened. No AI is involved.
"""
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROFILE = Path(__file__).resolve().parents[1] / 'efabric-profile'
HOME = 'https://efabric.cs.wisc.edu/workspace?tab=experiments/list'


def browser(p, headless):
    return p.chromium.launch_persistent_context(str(PROFILE), channel='chrome', headless=headless, viewport={'width': 1400, 'height': 1000})


def summary(page):
    """Compact, read-only description of what is on the page."""
    return page.evaluate('''() => {
        const vis = e => e.offsetParent !== null;
        const txt = e => (e.innerText || e.value || e.getAttribute('aria-label') || e.placeholder || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
        const out = {url: location.href, title: document.title};
        out.headings = [...document.querySelectorAll('h1,h2,h3,h4')].filter(vis).map(txt).filter(Boolean);
        out.buttons = [...document.querySelectorAll('button,[role=button],[role=tab],a')].filter(vis).map(txt).filter(Boolean);
        out.fields = [...document.querySelectorAll('input,select,textarea,[role=combobox]')].filter(vis).map(e => {
            const label = e.labels && e.labels[0] ? e.labels[0].innerText.trim() : '';
            return `${e.tagName.toLowerCase()} ${e.type || ''} label="${label}" name="${e.name || ''}" value="${(e.value || '').slice(0, 40)}" ph="${e.placeholder || ''}"`;
        });
        out.rows = [...document.querySelectorAll('tr,[role=row]')].filter(vis).map(txt).filter(Boolean).slice(0, 30);
        return out;
    }''')


GROUPS = ['NVIDIA-RTX-A5000', 'NVIDIA-A100-SXM4-40GB']  # preference order; never CPU
MEMORY_GB = 64  # our runs peak near 25 GiB of RAM; 64 for both GPU types
ACTIVE = re.compile(r'Running|Pending|Scheduled|Creating|Starting|Queued', re.I)


class NeedsYou(Exception):
    """Something a person must fix (login expired, no GPUs, page changed)."""


def main_text(page):
    return page.evaluate("() => (document.querySelector('main') || document.body).innerText")


def open_page(page, url):
    page.goto(url)
    page.wait_for_load_state('networkidle')
    page.wait_for_timeout(1500)
    if not page.url.startswith('https://efabric.cs.wisc.edu/workspace') or 'My Experiments' not in main_text(page):
        raise NeedsYou(f'EFabric login needed (landed on {page.url.split("?")[0]}); run: runs/pm/.venv/bin/python runs/pm/bin/efabric.py login')


def free_gpus(page):
    """{group: free GPU count} for our two GPU groups."""
    open_page(page, 'https://efabric.cs.wisc.edu/workspace?tab=overview%2Fcompute')
    free = {}
    for group in GROUPS:
        page.get_by_text(group, exact=True).first.click()
        page.wait_for_timeout(1500)
        text = ' '.join(main_text(page).split())
        match = re.search(re.escape(group) + r' \S+ · \d+GB (\d+) free GPUs', text)
        free[group] = int(match.group(1)) if match else 0
    return free


def workspaces(page):
    """{name: {'active': bool, 'ssh': 'user@host' or None, 'evicted_in': text}} from My Experiments."""
    open_page(page, HOME)
    for button in page.get_by_role('button', name='expand row').all():
        button.click()
        page.wait_for_timeout(500)
    text = ' '.join(main_text(page).split())
    found = {}
    for match in re.finditer(r'(ian-badgerscribe2?) [0-9a-f-]{36} (\S+)(.*?)(?= ian-badgerscribe2? [0-9a-f-]{36} | JOB NAME|$)', text):
        name, status, rest = match.groups()
        ssh = re.search(r'ssh -p 2200 (\S+@\d+\.\d+\.\d+\.\d+)', rest)
        evicted = re.search(r'Evicted in ((?:\d+h)?(?: ?\d+m)?)', rest)
        if name not in found or ACTIVE.search(status):  # an old stopped record must not hide a running one
            found[name] = {'active': bool(ACTIVE.search(status)), 'ssh': ssh.group(1) if ssh else None,
                           'evicted_in': evicted.group(1).strip() if evicted else None}
    return found


def create(page, name, group, dry_run=False):
    """Fill the four Create pages and click Create, but only after the preview shows exactly what we asked for."""
    open_page(page, 'https://efabric.cs.wisc.edu/workspace?tab=experiments%2Fcreate')
    page.get_by_role('button', name='Skip').click()  # 1) workflow image: default
    page.wait_for_timeout(1000)
    page.get_by_placeholder('my-workspace').fill(name)  # 2) name and group
    page.locator('[role=combobox]').first.click()
    page.get_by_role('option', name=group, exact=True).click()
    page.get_by_role('button', name='Next').click()
    page.wait_for_timeout(1000)
    memory = page.locator('input[type=number]').last  # 3) resources: defaults, memory per worker to 64
    memory.fill(str(MEMORY_GB))
    memory.press('Tab')
    page.get_by_role('button', name='Next').click()
    page.wait_for_timeout(1000)
    page.get_by_role('button', name='Validate').click()  # 4) no datasets
    create_button = page.get_by_role('button', name=re.compile(r'^Create', re.I))
    create_button.wait_for(timeout=60000)
    preview = main_text(page)
    preview = preview[preview.find('CONFIGURATION PREVIEW'):]
    want = {'workspace_label': name, 'group_name': group, 'gpu_count': '1', 'node_count': '1', 'memory_request_gb': str(MEMORY_GB)}
    got = {key: (re.search(rf'^{key}: (\S+)', preview, re.M) or [None, None])[1] for key in want}
    if got != want or 'gpu-base' not in preview:
        raise NeedsYou(f'Create page preview did not match, nothing created: {got}')
    if dry_run:
        return got
    create_button.click()
    page.wait_for_timeout(5000)


def renew(names, wait_minutes=20):
    """Create any of `names` that is not active, then wait for SSH commands. Returns {name: (user@host, group)}."""
    with sync_playwright() as p:
        context = browser(p, headless=True)
        page = context.new_page()
        try:
            missing = [n for n in names if not workspaces(page).get(n, {}).get('active')]
            groups = {}
            if missing:
                free = free_gpus(page)
                for name in missing:
                    group = next((g for g in GROUPS if free[g] > 0), None)
                    if group is None:
                        raise NeedsYou(f'no free A5000 or A100 GPUs right now ({free})')
                    create(page, name, group)
                    free[group] -= 1
                    groups[name] = group
            deadline = time.time() + wait_minutes * 60
            while True:
                found = workspaces(page)
                if all(found.get(n, {}).get('active') and found[n]['ssh'] for n in names):
                    return {n: (found[n]['ssh'], groups.get(n, 'already running'), found[n]['evicted_in']) for n in names}
                if time.time() > deadline:
                    raise NeedsYou(f'workspaces not ready after {wait_minutes} min: {found}')
                time.sleep(30)
        finally:
            context.close()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'inspect'
    if mode == 'renew':  # used by workspace_alert.py: one JSON line, exit 0, or "NEEDS YOU: ..." and exit 3
        try:
            print(json.dumps(renew(['ian-badgerscribe', 'ian-badgerscribe2'])))
        except NeedsYou as problem:
            print(f'NEEDS YOU: {problem}')
            sys.exit(3)
        return
    if mode == 'status':
        with sync_playwright() as p:
            context = browser(p, headless=True)
            page = context.new_page()
            print(free_gpus(page), workspaces(page))
            if len(sys.argv) > 2 and sys.argv[2] == '--dry-run':
                print('dry run, Create not clicked:', create(page, 'probe-never-created', GROUPS[0], dry_run=True))
            context.close()
        return
    with sync_playwright() as p:
        if mode == 'login':
            context = browser(p, headless=False)
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(HOME)
            print('Sign in to EFabric in the Chrome window that opened, then close that window.', flush=True)
            page.wait_for_event('close', timeout=0)
            context.close()
            return
        url = sys.argv[2] if len(sys.argv) > 2 else HOME
        assert url.startswith('https://efabric.cs.wisc.edu/'), url
        context = browser(p, headless=True)
        page = context.new_page()
        page.goto(url)
        page.wait_for_load_state('networkidle')
        for key, value in summary(page).items():
            print(f'{key}: {value}')
        context.close()


if __name__ == '__main__':
    main()
