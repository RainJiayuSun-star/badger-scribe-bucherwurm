const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {textContent:'', innerHTML:'', value:id==='filter'?'all':'', classList:{add(){},remove(){},toggle(){}}});
  return elements.get(id);
}
const context = vm.createContext({document:{getElementById:element,querySelectorAll:()=>[],addEventListener(){}},
  setInterval(){},fetch:async()=>({ok:false}),Date,Set,Math,console});
vm.runInContext(fs.readFileSync(__dirname+'/app.js','utf8'),context);
const fixture = {
  served_at:Date.now()/1000, refreshing:false,
  gpus:[{queue:'gpu1',stale:false,checked:new Date().toISOString(),gpu:{used:1024,total:24576,utilization:50},
    runs:[{id:'ft-test',unit:'gpu1',display_state:'active',status:{phase:'evaluating',epoch:2,eval_page:2,eval_pages:4},scores:[{cer:.1}]}]}],
  agents:{tasks:[{worker:'root',title:'<script>bad</script>',display_state:'completed',reported_at:Date.now()/1000,stage:'validation',source:'worker report'}]},
  badgerbrain:{reachable:true,key_loaded:true,key_accepted:true,latency_ms:64,models:['qwen3.8-27b'],usage:{spend:0.0263,max_budget:null},checked:Date.now()/1000,
    runs:[{id:'<b>screen</b>',model:'qwen3.8-27b',state:'complete',pages:94,total:94,hit_limit:4,retries:5,loop_cuts:6,wall_seconds:159,tokens:1000,updated:Date.now()/1000,cer_by_category:{survey_notes:0.2262}}]},
  models:[{name:'Churro',stage:'evaluation',state:'active',note:'Measured run',completed_runs:1,queued_runs:2}]
};
context.fixture=fixture;
vm.runInContext('render(fixture)',context);
assert.equal(element('active').textContent,'1 / 01');
assert.equal(element('fresh-agents').textContent,0);
assert.match(element('gpus').innerHTML,/EVAL PAGE 2 \/ 4/);
assert.match(element('workers').innerHTML,/&lt;script&gt;/);
assert.doesNotMatch(element('workers').innerHTML,/<script>bad/);
assert.match(element('badgerbrain').innerHTML,/ONLINE/);
assert.match(element('badgerbrain').innerHTML,/PAGE 94 \/ 94/);
assert.match(element('badgerbrain').innerHTML,/0\.2262/);
assert.doesNotMatch(element('badgerbrain').innerHTML,/<b>screen/);
vm.runInContext("$('badgerbrain').innerHTML=badgerbrainHtml({reachable:false,runs:[],models:[]})",context);
assert.match(element('badgerbrain').innerHTML,/Connect GlobalProtect/);
element('filter').value='complete';
vm.runInContext('drawRuns()',context);
assert.equal(element('runs').innerHTML,'');
vm.runInContext(fs.readFileSync(__dirname+'/training.js','utf8'),context);
vm.runInContext(`training={runs:[],public:{},overfit:[{name:'of1-<i>x</i>',model:'olmOCR-2-7B',epochs_to_use:2,
  held_mean:[[1,.3],[2,.25]],train_mean:[[1,.2],[2,.1]],
  runs:[{id:'of1-x-dominy_group1',fold:'dominy_group1',best_epoch:2,last_held:.25,last_train:.1,verdict:'no overfitting seen (best epoch 2)'}]}]}`,context);
const overfitHtml=vm.runInContext('overfitHtml()',context);
assert.match(overfitHtml,/epochs to use: <b>2<\/b>/);
assert.match(overfitHtml,/\+0\.1500/);
assert.match(overfitHtml,/Training-page CER \(mean\)/);
assert.doesNotMatch(overfitHtml,/<i>x/);
assert.match(vm.runInContext('training.overfit=[];overfitHtml()',context),/No overfitting tests/);
console.log('Interface rendering, escaping, progress and filtering checks passed');
