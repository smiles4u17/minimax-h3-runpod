const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js', 'utf8');
const fields = {
  samimate_prompt: 'Exact custom prompt', samimate_prompt_mode: 'custom',
  samimate_seed: '0', samimate_width: '1280', samimate_height: '720',
};
const selected = {steps: 8, sampler: 'euler', scheduler: 'simple', turbo_enabled: true,
  turbo_lora: 'chosen_ref2v_turbo.safetensors', ref2va_model: 'chosen.safetensors'};
const scope = {
  h3Settings: () => ({task: 'fl2v', steps: 20, sampler: 'h3_turbo', seed_random: true}),
  samimateH3Options: () => selected, payloadSettings: () => ({}),
  val: id => fields[id] || '', intFieldOrDefault: (id, fallback) => Number(fields[id] || fallback),
  samimateH3Duration: () => 5, samimateReferencePaths: () => ['identity.png'],
  samimateMaskRequestKey: () => 'current',
};
vm.createContext(scope);
for (const name of ['samimateH3Payload', 'samimatePreparedPayload', 'samimateAutoSize'])
  vm.runInContext(source.split('\n').find(s => s.startsWith(`function ${name}(`)), scope);
vm.runInContext('let SAMIMATE_MASKS=null', scope);
let p = scope.samimateH3Payload();
assert.equal(p.task, 'r2v');
assert.equal(p.steps, 8);
assert.equal(p.turbo_enabled, true);
assert.equal(p.ref2va_model, 'chosen.safetensors');
assert.equal(p.prompt_mode, 'custom');
assert.equal(p.prompt, fields.samimate_prompt);
assert.equal(p.seed, 0);
assert.equal(p.seed_random, false);
vm.runInContext(`SAMIMATE_MASKS={requestKey:'current',prepared:{source_path:'prepared.mkv',width:32,height:32},masks:{mask_video_path:'mask.mp4'}}`, scope);
p = scope.samimatePreparedPayload();
assert.equal(p.source_video_path, 'prepared.mkv');
assert.equal(p.mask_path, 'mask.mp4');
assert.equal(p.width, 1280);
vm.runInContext("SAMIMATE_MASKS.requestKey='stale'", scope);
assert.equal(scope.samimatePreparedPayload().mask_path, undefined);
console.log('SAMimate settings pass through; task remains Ref2V and stale masks are excluded');

const fullHD=scope.samimateAutoSize(1920,1080);
assert.ok(fullHD.width*fullHD.height<=2000000);
assert.equal(fullHD.width%32,0);
assert.equal(fullHD.height%32,0);
