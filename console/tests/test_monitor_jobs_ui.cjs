const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const elements={};
class Element {
  constructor(){this.children=[];this.classList={contains:()=>true}}
  setAttribute(name,value){this[name]=value}
  append(...items){this.children.push(...items)}
  appendChild(item){this.append(item)}
  replaceChildren(){this.children=[]}
  querySelector(){return {after(){}}}
  set innerHTML(value){for(const id of ['monitor_requests_refresh','monitor_request_status','monitor_request_list'])elements[id]=new Element()}
}
elements.monitor=new Element();
let resolveCancel,rejectCancel,selected='other-job';const posts=[];
const rows=[{endpoint_id:'ep',job_id:'first',status:'IN_QUEUE',can_cancel:true},
            {endpoint_id:'ep',job_id:'done',status:'COMPLETED',can_cancel:false}];
const scope={document:{getElementById:id=>elements[id],createElement:()=>new Element(),hidden:false},
  window:{showTab(){},attachJobMonitor:(ep,job)=>selected=job},setInterval(){},encodeURIComponent,Date,
  api:async(path,options)=>{if(!options)return {items:rows.map(j=>({...j}))};posts.push(path);return new Promise((resolve,reject)=>{resolveCancel=resolve;rejectCancel=reject})}};
vm.createContext(scope);vm.runInContext(fs.readFileSync('static/monitor_jobs.js','utf8'),scope);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const cancel=()=>elements.monitor_request_list.children[0].children[2].children[1];
(async()=>{
  await flush();assert.equal(elements.monitor_request_list.children[1].children[2].children[1].disabled,true);
  const first=cancel(),pending=first.onclick();first.onclick();assert.equal(posts.length,1);
  scope.window.attachJobMonitor('ep','different-job');assert.equal(selected,'different-job');
  rejectCancel(new Error('network'));await pending;await flush();assert.equal(cancel().disabled,false);
  const retry=cancel().onclick();resolveCancel({message:'requested'});await retry;await flush();
  assert.equal(posts[1],'/api/job/ep/first/cancel');assert.equal(cancel().disabled,true);
  cancel().onclick();assert.equal(posts.length,2);
  console.log('Per-row targeting, duplicate suppression, terminal disabling and failure retry passed');
})();
