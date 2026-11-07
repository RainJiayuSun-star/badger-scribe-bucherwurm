const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {textContent:'', innerHTML:'', value:id==='filter'?'all':'', classList:{add(){},remove(){},toggle(){}}});
  return elements.get(id);
}
const context = vm.createContext({document:{getElementById:element,querySelectorAll:()=>[]},
  setInterval(){},fetch:async()=>({ok:false}),Date,Set,Math,console});
vm.runInContext(fs.readFileSync(__dirname+'/app.js','utf8'),context);
const fixture = {
  served_at:Date.now()/1000, refreshing:false,
  gpus:[{queue:'gpu1',stale:false,checked:new Date().toISOString(),gpu:{used:1024,total:24576,utilization:50},
    runs:[{id:'ft-test',unit:'gpu1',display_state:'active',status:{phase:'evaluating',epoch:2,eval_page:2,eval_pages:4},scores:[{cer:.1}]}]}],
  agents:{tasks:[{worker:'root',title:'<script>bad</script>',display_state:'completed',reported_at:Date.now()/1000,stage:'validation',source:'worker report'}]},
  models:[{name:'Churro',stage:'evaluation',state:'active',note:'Measured run',completed_runs:1,queued_runs:2}]
};
context.fixture=fixture;
vm.runInContext('render(fixture)',context);
assert.equal(element('active').textContent,'1 / 02');
assert.equal(element('fresh-agents').textContent,0);
assert.match(element('gpus').innerHTML,/EVAL PAGE 2 \/ 4/);
assert.match(element('workers').innerHTML,/&lt;script&gt;/);
assert.doesNotMatch(element('workers').innerHTML,/<script>bad/);
element('filter').value='complete';
vm.runInContext('drawRuns()',context);
assert.equal(element('runs').innerHTML,'');
console.log('Interface rendering, escaping, progress and filtering checks passed');
