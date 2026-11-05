const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('static/h3_variants.js','utf8');
const values={h3_task:'r2v',h3_steps:'8',h3_second_pass_sigma:'2'},checked={h3_no_turbo:true,h3_latent_upscale:true};
const elements={};
function node(){return {hidden:false,classList:{add(){},toggle(name,on){this[name]=on}},closest(){return this},refreshPreview(){}}}
const get=id=>elements[id] ||=node();
get('h3_reference_grid').previousElementSibling=node();get('h3_reference_grid').previousElementSibling.previousElementSibling=node();
const ctx={$:get,val:id=>values[id]||'',chk:id=>!!checked[id],set:(id,v)=>values[id]=String(v),setChk:(id,v)=>checked[id]=v,
 isNew:()=>true,variant:()=>values.h3_task==='r2v'?'ref2v_20260920':'fflf_20260920',photos:node(),
 document:{querySelector:()=>node()},updateH3TaskUI(){},updateH3TurboUI(){},persistSettingsSoon(){},h3Settings:()=>({steps:8})};
ctx.window=ctx;vm.createContext(ctx);
vm.runInContext(source.slice(source.indexOf('  function sync('),source.indexOf("  $('h3_task').addEventListener")),ctx);
vm.runInContext('sync()',ctx);
assert.equal(get('h3_photo9').classList.hidden,false);
assert.equal(get('h3_keyframe9').classList.hidden,true);
for(let i=1;i<=9;i++)values['h3_photo'+i]='view-'+i+'.png';
vm.runInContext(source.slice(source.indexOf('  const original=h3Settings;'),source.indexOf('  const apply=applyH3Settings;')),ctx);
let settings=ctx.h3Settings();
assert.equal(settings.photo_paths.length,9);assert.equal(settings.reference_paths[8],'view-9.png');
values.h3_photo2='';settings=ctx.h3Settings();assert.equal(settings.reference_paths[1],'view-3.png');
values.h3_task='fl2v';vm.runInContext('sync()',ctx);
assert.equal(get('h3_photo9').classList.hidden,true);assert.equal(ctx.h3Settings().photo_paths.length,4);
values.h3_task='r2v';checked.h3_use_multi_image=true;vm.runInContext('sync()',ctx);
assert.equal(get('h3_photo9').classList.hidden,true);assert.equal(ctx.h3Settings().photo_paths.length,4);
checked.h3_use_multi_image=false;vm.runInContext('sync()',ctx);assert.equal(get('h3_photo9').classList.hidden,false);
console.log('Nine Ref2V uploads and ordered payload; four-slot FL2V/keyframes preserved');
