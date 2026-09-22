const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const els={h3_cancel_button:{disabled:true},h3_status:{}},stored=new Map(),requests=[];
let resolve,reject;
const scope={document:{getElementById:id=>els[id]},localStorage:{setItem:(k,v)=>stored.set(k,v),getItem:k=>stored.get(k),removeItem:k=>stored.delete(k)},window:{addEventListener(){}},log(){},encodeURIComponent,api:(path,options)=>{requests.push({path,options});return new Promise((a,b)=>{resolve=a;reject=b})}};
vm.createContext(scope);vm.runInContext(fs.readFileSync('static/h3_job.js','utf8'),scope);
(async()=>{
const w=scope.window;w.trackCurrentH3Job('endpoint','job1');assert.equal(els.h3_cancel_button.disabled,false);
const pending=w.cancelCurrentH3Job();assert.equal(els.h3_cancel_button.disabled,true);await w.cancelCurrentH3Job();assert.equal(requests.length,1);
w.trackCurrentH3Job('endpoint','job2');resolve({message:'requested'});await pending;assert.equal(els.h3_cancel_button.disabled,false);
w.finishCurrentH3Job('endpoint','job1');assert.equal(els.h3_cancel_button.disabled,false);
const failed=w.cancelCurrentH3Job();reject(new Error('network'));await failed;assert.equal(els.h3_cancel_button.disabled,false);
const retry=w.cancelCurrentH3Job();resolve({});await retry;assert.equal(els.h3_cancel_button.disabled,true);
assert.equal(requests[2].path,'/api/job/endpoint/job2/cancel');assert.equal(requests[2].options.method,'POST');
w.finishCurrentH3Job('endpoint','job2');assert.equal(els.h3_cancel_button.disabled,true);assert.equal(stored.size,0);
console.log('H3 cancel targets, duplicate suppression, retry and stale-job races passed');
})();
