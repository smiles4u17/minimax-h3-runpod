const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const values = {h3_task:'fl2v', h3_sampler:'lcm', h3_scheduler:'simple', h3_steps:'8',
 h3_pass1_split:'8', h3_second_pass_sigma:'5', h3_attention:'sage', h3_attention_profile:'custom',
 h3_fl2va_model:'my-hybrid.safetensors', h3_loras:'[]', h3_acceleration_recipe:'custom'};
const checked = {h3_cache_enabled:false, h3_latent_upscale:true, h3_no_turbo:true,
 h3_use_larry:true, h3_sparse_trained_weights:false};
const elements = {};
for (const id of [...Object.keys(values), ...Object.keys(checked), 'h3_acceleration_status'])
 elements[id] = {type:id in checked?'checkbox':'text', options:[], dispatchEvent(){}, add(option){this.options.push(option)}};
const scope = {window:{h3Settings:()=>({sampler:values.h3_sampler}),pbGenerationSettings:()=>({}),applyH3Settings(){},addEventListener(){}},
 localStorage:{getItem(){return null},setItem(){}}, $:id=>elements[id],val:id=>values[id]||'',chk:id=>!!checked[id],
 set:(id,v)=>values[id]=String(v),setChk:(id,v)=>checked[id]=v,persistSettingsSoon(){},pbSaveGeneration(){},renderH3LoraActive(){},
 h3Loras:()=>JSON.parse(values.h3_loras),setH3Loras:items=>values.h3_loras=JSON.stringify(items),
 Event:class {},Option:class {constructor(text,value){this.value=value}}};
vm.createContext(scope);
let source=fs.readFileSync('static/h3_acceleration.js','utf8').replace('  function install(prefix, saved) {','  window.choose = choose;\n  function install(prefix, saved) {');
vm.runInContext(source,scope);
(async()=>{
 assert.equal(values.h3_sampler,'lcm');
 assert.equal(scope.window.h3Settings().acceleration_recipe,'custom');
 values.h3_acceleration_recipe='sla4'; await scope.window.choose('h3');
 assert.equal(values.h3_sampler,'euler'); assert.equal(JSON.parse(values.h3_loras).length,1);
 values.h3_acceleration_recipe='pdd8'; await scope.window.choose('h3');
 assert.equal(JSON.parse(values.h3_loras).length,0, 'Switching recipes must not stack SLA weights onto PDD');
 assert.equal(values.h3_steps,'8');
 values.h3_acceleration_recipe='custom'; await scope.window.choose('h3');
 assert.equal(values.h3_sampler,'lcm'); assert.equal(values.h3_fl2va_model,'my-hybrid.safetensors');
 assert.equal(checked.h3_use_larry,true);
 console.log('LCM/custom settings preserved; recipe switching restores adapters and model');
})().catch(error=>{console.error(error);process.exitCode=1});
