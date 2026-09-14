const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js', 'utf8');
const fields = {samimate_image_path:'front.png', samimate_reference3_path:'side.png', samimate_prompt:'red jacket',
  samimate_subject_prompt:'woman on left', samimate_video_path:'source.mp4', samimate_seed:'0',
  samimate_width:'832',samimate_height:'480',samimate_video_start:'2',samimate_video_end:'7'};
const sandbox = {val:id=>fields[id]||'', h3Settings:()=>({task:'fl2v',prompt:'stale',reference_paths:['unrelated.png'],turbo_enabled:true}),
  samimateH3Options:()=>({turbo_enabled:false,steps:20,sampler:'res_multistep'}),
  payloadSettings:()=>({}), videoDuration:()=>20, timeVal:(id,d)=>Number(fields[id]||d),
  samimateWanFrameCap:()=>0, intFieldOrDefault:(id,d)=>Number(fields[id]||d)};
vm.createContext(sandbox);
for (const name of ['samimateReferencePaths','samimateH3Duration','samimateH3Payload']) {
  vm.runInContext(source.split('\n').find(line=>line.startsWith(`function ${name}(`)),sandbox);
}
const payload = vm.runInContext('samimateH3Payload()',sandbox);
assert.deepEqual(Array.from(payload.reference_paths),['front.png','side.png']);
assert.equal(payload.duration,5);
assert.equal(payload.seed,0);
assert.equal(payload.task,'r2v');
assert.equal(payload.turbo_enabled,false);
assert.equal(payload.steps,20);
assert.equal(payload.prompt,'red jacket');
assert.equal(payload.reference_video_paths.length,0);
fields.samimate_video_end='19';
assert.equal(vm.runInContext('samimateH3Duration()',sandbox),17,'Never silently cap the requested duration');
console.log('SAMimate UI payload checks passed');
