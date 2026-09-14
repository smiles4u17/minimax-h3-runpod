const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js','utf8');
const values = {h3_task:'fl2v',h3_steps:'160',h3_turbo_enabled:true,h3_turbo_lora:'minimax_h3_turbo_v4_step600_ema.safetensors'};
const elements = {};
for (const id of ['h3_steps','h3_turbo_lora','h3_turbo_family','h3_turbo_strength','h3_cache_threshold','h3_turbo_notice'])
  elements[id]={options:[],value:values[id],removeAttribute(name){delete this[name]}};
elements.h3_steps.max='8';
const scope = {$:id=>elements[id],val:id=>values[id]||'',chk:id=>!!values[id],
  set:(id,value)=>{values[id]=value;if(elements[id])elements[id].value=value},
  fillH3ModelSelect:(id,items)=>elements[id].options=items.map(value=>({value})),
  updateH3QualityNotice(){},persistSettingsSoon(){}};
vm.createContext(scope);
for (const name of ['updateH3TurboUI','selectH3TurboFamily'])
  vm.runInContext(source.split('\n').find(s=>s.startsWith(`function ${name}(`)),scope);
vm.runInContext("selectH3TurboFamily('lightx2v')",scope);
assert.match(values.h3_turbo_lora, /fl2v_turbo_4step.*comfyui/);
assert.equal(values.h3_sampler,'euler');
assert.equal(elements.h3_steps.value,'160');
assert.equal(elements.h3_steps.max,undefined);
values.h3_task='r2v';
vm.runInContext("selectH3TurboFamily('lightx2v')",scope);
assert.match(values.h3_turbo_lora,/ref2v_turbo/);
values.h3_turbo_lora='H3/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors';
vm.runInContext("selectH3TurboFamily('lightx2v')",scope);
assert.equal(values.h3_turbo_lora,'H3/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors');
vm.runInContext("selectH3TurboFamily('larry')",scope);
assert.match(values.h3_turbo_lora,/v4_step600_ema/);
assert.equal(values.h3_sampler,'h3_turbo');
assert.equal(elements.h3_steps.value,'160');
console.log('Turbo switching preserves steps and selects task-matched weights');
