'use strict';
let snapshot = null;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let names = {root:'Coordinator'};  // display names; each developer adds their own agents in config.json
function state(r) { return r.display_state || 'unknown'; }
function badge(s, label) {
  const color = ['complete','completed'].includes(s) ? 'good' : ['failed','blocked','unknown'].includes(s) ? 'bad' : ['queued','waiting','idle'].includes(s) ? 'dim' : '';
  return `<span class="badge ${color}">${esc(label || s)}</span>`;
}
function time(s) {
  return s ? new Date(typeof s === 'number' ? s * 1000 : s).toLocaleString('en-US', {timeZone:'America/Chicago',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'}) + ' CST' : 'No observation';
}
function duration(seconds) {
  if (!(seconds >= 0)) return '—';
  const h = Math.floor(seconds / 3600), m = Math.floor(seconds % 3600 / 60), s = Math.floor(seconds % 60);
  return `${h}h ${String(m).padStart(2,'0')}m ${String(s).padStart(2,'0')}s`;
}
function timer(iso, label, mode='since') {
  return `<div><b ${iso?`data-${mode}="${esc(iso)}"`:''}>—</b><small>${esc(label)}</small></div>`;
}
function updateTimers() {
  const now = Date.now();
  document.querySelectorAll('[data-since]').forEach(el => el.textContent = duration((now - Date.parse(el.dataset.since)) / 1000));
  document.querySelectorAll('[data-until]').forEach(el => el.textContent = duration((Date.parse(el.dataset.until) - now) / 1000));
}
function gpu(g) {
  const r = g.runs.find(r => ['active','unknown'].includes(state(r))), hw = g.gpu || {}, s = r?.status || {};
  const evaluating = s.phase === 'evaluating';
  const current = evaluating ? s.eval_page : s.step, total = evaluating ? s.eval_pages : s.total_steps;
  const percent = total ? Math.min(100, 100 * (current || 0) / total) : null;
  const stale = g.stale || state(r || {}) === 'unknown';
  return `<article class="gpu"><div class="gpuhead"><span class="unit">${esc(g.queue.toUpperCase())} / A5000</span>${badge(g.stale?'unknown':r?state(r):'queued',g.stale?'STALE / OFFLINE':r?(state(r)==='unknown'?'ACTIVITY UNCONFIRMED':s.phase):'NO ACTIVE RUN')}</div>
    <div class="runname">${esc(r?.id || 'No job running')}</div><div class="gpumeta">${r?'Epoch '+esc(s.epoch??'—')+' · '+esc(s.phase):''}${r&&stale?' · last known state':''}</div>
    <div class="progress"><span style="width:${percent??0}%;${stale?'background:var(--muted)':''}"></span></div><div class="pair"><span>${total?(evaluating?'EVAL PAGE ':'STEP ')+esc(current||0)+' / '+esc(total):'Progress not reported'}</span><span>${percent===null?'':Math.floor(percent)+'%'}</span></div>
    <div class="hardware"><div><b>${hw.utilization??'—'}%</b><small>GPU UTILIZATION</small></div><div><b>${hw.used?(hw.used/1024).toFixed(1):'—'} / ${hw.total?(hw.total/1024).toFixed(0):'—'}</b><small>MEMORY · GiB</small></div><div><b>${g.runs.filter(r=>state(r)==='queued').length}</b><small>QUEUED</small></div></div>
    <div class="hardware">${timer(r&&s.started,'JOB RUNNING')}${timer(r&&r.epoch_started,'EPOCH '+(s.epoch??'—')+' RUNNING')}${timer(g.workspace_started,'WORKSPACE UP')}${timer(g.workspace_started&&new Date(Date.parse(g.workspace_started)+864e5).toISOString(),'UNTIL 24-HOUR STOP','until')}</div>
    <div class="source">CHECKED ${time(g.checked)} · RUN LAST UPDATED ${time(s.updated)}</div>${g.error?'<p class="warn">'+esc(g.error)+'</p>':''}</article>`;
}
function model(m) {
  const stages=['development','validation','fine-tuning','evaluation','release'];
  return `<article class="model"><div class="modeltitle"><h3>${esc(m.name)}</h3>${badge(m.state,m.state==='active'?m.stage:m.state)}</div><div class="stages">${stages.map(stage=>`<div class="stage ${m.state==='active'&&m.stage.includes(stage)?'current':''}">${esc(stage.toUpperCase())}</div>`).join('')}</div><div class="modelnote">${esc(m.note)}</div><div class="source mono" style="margin-top:12px;color:var(--muted);font-size:10px">${m.completed_runs} COMPLETED RUNS · ${m.queued_runs} QUEUED</div></article>`;
}
function task(t) {
  return `<div class="task ${t.display_state==='completed'?'complete':''}"><strong>${esc(t.title)}</strong>${t.note?'<div>'+esc(t.note)+'</div>':''}<div style="font:10px monospace;margin-top:5px;color:var(--muted)">${esc(t.stage)} · ${time(t.reported_at)}<br>${esc(t.source)}${t.display_state==='unknown'?' · REPORT EXPIRED':''}</div></div>`;
}
function group(label, tasks, css='') {
  return `<div class="taskhead ${css}">${label} / ${tasks.length}</div>${tasks.length?tasks.map(task).join(''):'<div class="none">None reported</div>'}`;
}
function worker(handle, tasks) {
  const active=tasks.filter(t=>t.display_state==='active'), waiting=tasks.filter(t=>t.display_state==='waiting'), unknown=tasks.filter(t=>['blocked','unknown'].includes(t.display_state)), complete=tasks.filter(t=>t.display_state==='completed');
  const status=active.length?'active':unknown.some(t=>t.display_state==='unknown')?'unknown':unknown.length?'blocked':waiting.length?'waiting':'idle';
  return `<article class="worker"><div class="gpuhead"><span class="num">${handle==='root'?'CONTROL':'WORKER'}</span>${badge(status)}</div><h3>${esc(names[handle] || handle)}</h3><div class="handle">${esc(handle)}</div><p class="mission">Last report: ${time(Math.max(...tasks.map(t=>t.reported_at)))}</p>${group('IN PROGRESS',active)}${unknown.length?group('BLOCKED / UNCONFIRMED',unknown):''}${group('WAITING',waiting)}<details data-worker="${esc(handle)}"><summary>COMPLETED / ${complete.length}</summary>${complete.map(task).join('')}</details></article>`;
}
// Each developer maps their agents to a role under "roles" in config.json; unmapped workers show under "Role not set".
const ROLES=[['code','Code workers','One feature on its own branch and worktree, with tests'],
  ['analysis','Analysis workers','Paired comparisons by the plan\'s rules; numbers only'],
  ['review','Review workers','Code review before Ian merges a PR']];
const unfinished=t=>!['completed','cancelled'].includes(t.display_state);
function hierarchy(d, tasks) {
  const of=w=>tasks.filter(t=>t.worker===w), ROLE_OF=d.roles||{};
  const root=of('root'), active=root.filter(t=>t.display_state==='active');
  const coordState=active.length?'active':root.some(t=>t.display_state==='unknown')?'unknown':'idle';
  const workers=[...new Set(tasks.filter(t=>!['root','ian'].includes(t.worker)&&unfinished(t)).map(t=>t.worker))].sort();
  const roleCol=(key,label,duty)=>{const ws=workers.filter(w=>(ROLE_OF[w]||'unset')===key);
    return `<div class="rolecol"><div class="rolehead"><b>${esc(label)}</b><small>${esc(duty)}</small></div>${ws.length?ws.map(w=>worker(w,of(w))).join(''):'<div class="none">None running. Started per task.</div>'}</div>`};
  const unset=workers.some(w=>!ROLE_OF[w]);
  const queues=d.gpus.map(g=>{const r=g.runs.find(r=>state(r)==='active');return `<div class="qchip"><b>${esc(g.queue.toUpperCase())}</b> ${esc(r?r.id:'no job running')}<br><small>${g.runs.filter(r=>state(r)==='queued').length} queued</small></div>`}).join('')||'<div class="none">No workspace data yet.</div>';
  return `<div class="tier"><div class="tierlabel">Coordinates</div><article class="node coord"><div class="gpuhead"><h3>Coordinator</h3>${badge(coordState)}</div><p class="duty">Restarted every 30 to 45 minutes from STATE.md. Owns STATE.md, the experiment log and the queue files. Judges results by the plan's rules and queues the next job.</p>${active.length?group('IN PROGRESS',active):'<div class="none">No active coordinator task.</div>'}<details data-worker="root"><summary>COMPLETED / ${root.filter(t=>t.display_state==='completed').length}</summary>${root.filter(t=>t.display_state==='completed').map(task).join('')}</details></article></div>
  <div class="tier"><div class="tierlabel">Workers, started per task</div><div class="roles">${ROLES.map(r=>roleCol(...r)).join('')}${unset?roleCol('unset','Role not set','Add this worker under "roles" in config.json'):''}</div></div>
  <div class="tier last"><div class="tierlabel">Runs the GPU jobs, no agent</div><div class="queues">${queues}</div></div>`;
}
function render(d) {
  snapshot=d;
  names={root:'Coordinator',...(d.names||{})};
  const runs=d.gpus.flatMap(g=>g.runs), tasks=d.agents.tasks;
  $('active').textContent=d.gpus.filter(g=>!g.stale&&g.runs.some(r=>state(r)==='active')).length+' / 02';
  $('finished').textContent=runs.filter(r=>state(r)==='complete').length;
  $('waiting').textContent=tasks.filter(t=>['waiting','blocked'].includes(t.display_state)).length;
  $('fresh-agents').textContent=new Set(tasks.filter(t=>t.display_state==='active').map(t=>t.worker)).size;
  $('gpus').innerHTML=d.gpus.length?d.gpus.map(gpu).join(''):'<div class="loading">Connecting to the GPU workspaces…</div>';
  $('models').innerHTML=d.models.map(model).join('');
  const open = new Set([...document.querySelectorAll('details[open]')].map(x=>x.dataset.worker));
  $('workers').innerHTML=hierarchy(d,tasks);
  document.querySelectorAll('details').forEach(x=>x.open=open.has(x.dataset.worker));
  $('agentnote').textContent='A task counts as active only if its worker reported in the last 5 minutes. Workers whose tasks are all finished are hidden.';
  $('refresh').textContent=d.refreshing?'↻ REFRESHING…':'↻ REFRESH';
  $('transport').textContent='UPDATED '+time(d.served_at);
  drawRuns();
  updateTimers();
  $('todo').innerHTML=todoHtml();
  const sub=submissionBox(d.submissions||[]);
  $('subbox').innerHTML=sub.html;$('subbox').className='subbox '+sub.cls;
  if($('waitlist').open)fillWaiting();
}
function todoHtml() {
  const tasks=(snapshot?.agents.tasks||[]).filter(t=>['waiting','blocked'].includes(t.display_state));
  return tasks.length?tasks.map(t=>`<div class="task"><strong>${esc(t.title)}</strong> ${badge(t.display_state)}<div>${esc(names[t.worker]||t.worker)} · ${esc(t.stage)} · reported ${time(t.reported_at)}</div>${t.note?'<div>'+esc(t.note)+'</div>':''}</div>`).join(''):'<div class="none">Nothing is awaiting action.</div>';
}
function submissionBox(subs) {
  const ready=subs.filter(s=>s.csv&&!s.uploaded), done=subs.filter(s=>s.uploaded);
  const last=done[done.length-1];
  const lastLine=last?`<p>Last uploaded: <b>${esc(last.id)}</b>${last.public_score!=null?' · public score <b>'+Number(last.public_score).toFixed(3)+'</b>':' · public score not recorded yet'}</p>`:'<p>Nothing uploaded yet.</p>';
  if(!ready.length) return {cls:'', html:`<h3>No new submission to upload</h3>${lastLine}`};
  const rows=ready.map(s=>{const problems=[s.on_mac?'':'not copied to the Mac yet',s.pages!=null&&s.pages!==208?s.pages+' of 208 pages':'',s.empty_pages?s.empty_pages+' empty pages':'',s.dirty?'recorded a modified commit, so it can\'t be a final entry':''].filter(Boolean);
    return `<p><b>${esc(s.id)}</b> · finished ${time(s.finished_at)} · ${s.pages??'?'} pages · commit ${esc(s.commit||'unknown')}${problems.length?' · <span style="color:var(--red)">'+esc(problems.join(', '))+'</span>':''}</p>`}).join('');
  return {cls:'ready', html:`<h3>Ready to upload: ${ready.map(s=>esc(s.id)).join(', ')}</h3>${rows}<p>File: runs/finetune/&lt;id&gt;/submission.csv. After uploading, add the id to "submitted" in runs/ops-dashboard/history_notes.json, and its score to "public_scores" once Kaggle shows it. Kaggle allows 2 uploads a day.</p>${lastLine}`};
}
function fillWaiting() { $('waitlist-body').innerHTML=todoHtml(); }
function openWaiting(){fillWaiting();$('waitlist').showModal();}
function drawRuns() {
  if(!snapshot)return;
  const q=$('search').value.toLowerCase(),filter=$('filter').value;
  const rows=snapshot.gpus.flatMap(g=>g.runs).filter(r=>r.id.toLowerCase().includes(q)&&(filter==='all'||state(r)===filter));
  $('runs').innerHTML=rows.map(r=>`<tr><td class="id">${esc(r.id)}</td><td>${esc(r.unit.toUpperCase())}</td><td>${badge(state(r),state(r)==='active'?r.status.phase:state(r))}</td><td>${esc(r.status.epoch??'—')}</td><td class="mono">${r.scores.length?Math.min(...r.scores.map(s=>s.cer)).toFixed(4):'—'}</td></tr>`).join('');
  $('empty').classList.toggle('hidden',rows.length>0);
}
async function load() {
  try {
    const r=await fetch('/api/status');if(!r.ok)throw Error();
    render(await r.json());$('connection').classList.add('hidden');
  } catch(e) {
    $('connection').textContent='The dashboard server is not responding. Showing the last data received.';
    $('connection').classList.remove('hidden');
    $('active').textContent='— / 02';$('fresh-agents').textContent='—';
  }
}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('on',x===b));
  document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('hidden',x.dataset.view!==b.dataset.tab));
});
$('waiting-card').onclick=openWaiting;
$('waiting-card').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openWaiting();}};
$('waitlist-close').onclick=()=>$('waitlist').close();
$('search').oninput=drawRuns;$('filter').onchange=drawRuns;
$('refresh').onclick=async()=>{try{const r=await fetch('/api/refresh',{method:'POST'});if(!r.ok)throw Error();$('refresh').textContent='↻ REFRESHING…';await load()}catch(e){load()}};
function tick(){$('clock').textContent=new Date().toLocaleString('en-US',{timeZone:'America/Chicago',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false})+' CST';updateTimers()}
tick();setInterval(tick,1000);load();setInterval(load,5000);
