const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = {};
function element(value = '') {
  return {value, options: [], listeners: {}, append(option) {this.options.push(option)},
    addEventListener(event, fn) {this.listeners[event] = fn},
    closest() {return {after() {}}},
    set innerHTML(html) {
      for (const [, id] of html.matchAll(/id="([^"]+)"/g)) elements[id] = element();
    }};
}
elements.pb_attention = element('sage');
elements.pb_attention.options = [{value:'sage'}, {value:'auto'}, {value:'native'}];
elements.pb_cache_enabled = {checked:true};
let ready, stored = {}, saves = 0;
const catalog = {cards:{}, note:'Allocated worker', profiles:{
  custom:{label:'Custom',attention:null}, sage:{label:'Sage',attention:'sage'},
  sol:{label:'Sol',attention:'sol-attn'}, sla:{label:'SLA',attention:'sla'},
  vsa:{label:'VSA',attention:'vsa',disabled:true}}};
const context = {document:{getElementById:id=>elements[id],createElement:()=>element(),
  querySelector:selector=>elements.pb_attention_profile?.options.find(option=>selector.includes(`"${option.value}"`))},
  h3Settings:()=>({}),pbGenerationSettings:()=>({attention:elements.pb_attention.value}),
  applyH3Settings(){},SETTINGS:{},api:async()=>catalog,log:message=>{throw Error(message)},
  localStorage:{getItem:()=>JSON.stringify(stored)},
  addEventListener:(event,fn)=>ready=fn,
  pbSaveGeneration() {saves++;stored={attention_profile:elements.pb_attention_profile.value,attention:elements.pb_attention.value}}};
context.window=context;
vm.createContext(context);
const source=fs.readFileSync(require('node:path').join(__dirname,'../static/h3_attention_profiles.js'),'utf8');
vm.runInContext(source,context);
(async()=>{
  await ready(); // Installation occurs after the ordinary Prompt Book field bindings.
  const profiles=elements.pb_attention_profile;
  profiles.value='sol';profiles.onchange();
  assert.equal(elements.pb_attention.value,'sol-attn');
  assert.ok(elements.pb_attention.options.some(option=>option.value==='sol-attn'));
  assert.equal(elements.pb_cache_enabled.checked,false);
  assert.equal(stored.attention_profile,'sol');
  assert.equal(context.pbGenerationSettings().attention,'sol-attn');
  assert.equal(context.pbGenerationSettings().attention_profile,'sol');
  delete elements.pb_attention_profile;
  await ready();
  assert.equal(elements.pb_attention_profile.value,'sol','Restore saved profile after asynchronous installation');
  elements.pb_attention.value='native';elements.pb_attention.listeners.change();
  assert.equal(stored.attention_profile,'custom','Manual changes clear and save profile');
  assert.equal(stored.attention,'native');
  assert.equal(saves,2);
  assert.equal(elements.pb_attention_profile.options.find(option=>option.value==='sla').disabled,true);
  assert.equal(elements.pb_attention_profile.options.find(option=>option.value==='vsa').disabled,true);
  console.log('Prompt Book attention selection, persistence and compatibility passed');
})().catch(error=>{console.error(error);process.exitCode=1});
