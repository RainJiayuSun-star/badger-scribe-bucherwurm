"""Per-developer settings from config.json, which git ignores. Start from config.example.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    path = HERE / 'config.json'
    if not path.exists():
        raise SystemExit(f'Missing {path}. Copy config.example.json to config.json and fill in your workspaces.')
    config = json.loads(path.read_text())
    for workspace in config['workspaces']:
        workspace.setdefault('host', config.get('ssh_host'))
        workspace.setdefault('port', config.get('ssh_port', 22))
    return config


def ssh(workspace, command, timeout):
    """Argument list for a read-only, non-interactive ssh call to one workspace."""
    return ['ssh', '-n', '-p', str(workspace['port']), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            f"{workspace['user']}@{workspace['host']}", command]
