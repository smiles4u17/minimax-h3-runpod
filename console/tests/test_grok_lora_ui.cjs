const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const app = fs.readFileSync('static/app.js', 'utf8');
const values = {h3_loras: JSON.stringify([{name: 'H3/robot.safetensors', strength: .65}]),
  pb_loras: '[]', h3_prompt: 'A robot walks.', grok_model: 'grok-4.7'};
const elements = {h3_enhance_button: {disabled: false}, h3_enhance_undo: {disabled: true},
  h3_enhance_status: {}, h3_prompt: {dispatchEvent() {}}};
const scope = {$: id => elements[id], val: id => values[id] || '',
  set: (id, value) => { values[id] = value; },
  persistSettingsQuietly() {}, persistSettingsSoon() {}, renderH3LoraActive() {},
  document: {readyState: 'loading', addEventListener() {}},
  localStorage: {getItem() { return this.saved || null; }, setItem(key, value) { this.saved = value; }},
  Event: function Event() {}, alert(message) { throw new Error(message); }, log() {}};
vm.createContext(scope);
for (const name of ['h3Loras', 'setH3Loras', 'toggleH3Lora', 'updateH3LoraStrength', 'removeH3LoraFromWorkflow'])
  vm.runInContext(app.split('\n').find(line => line.startsWith(`function ${name}(`)), scope);
vm.runInContext('toggleH3Lora(0,false)', scope);
assert.equal(JSON.parse(values.h3_loras)[0].enabled, false);
vm.runInContext('updateH3LoraStrength(0,0.8); toggleH3Lora(0,true)', scope);
assert.deepEqual(JSON.parse(values.h3_loras)[0], {name: 'H3/robot.safetensors', strength: .8, enabled: true});
vm.runInContext('removeH3LoraFromWorkflow(0)', scope);
assert.deepEqual(JSON.parse(values.h3_loras), []);
vm.runInContext(fs.readFileSync('static/prompt_book.js', 'utf8'), scope);
values.pb_lora_library_select = 'H3/robot.safetensors'; values.pb_lora_strength = '0.7';
vm.runInContext('pbAddLora()', scope);
assert.equal(JSON.parse(scope.localStorage.saved).loras[0].enabled, true);
values.pb_loras = JSON.stringify([{name: 'H3/robot.safetensors', strength: .7, enabled: false}]);
vm.runInContext('pbSaveGeneration()', scope);
values.pb_loras = '[]';
vm.runInContext('pbRestoreGeneration()', scope);
assert.equal(JSON.parse(values.pb_loras)[0].enabled, false);
vm.runInContext('pbRemoveLora(0)', scope);
assert.deepEqual(JSON.parse(scope.localStorage.saved).loras, []);
vm.runInContext(fs.readFileSync('static/grok_enhancer.js', 'utf8'), scope);
(async () => {
  scope.api = async () => ({prompt: 'A robot walks in soft light.', model: 'grok-4.7'});
  await vm.runInContext('enhanceH3Prompt()', scope);
  assert.equal(values.h3_prompt, 'A robot walks in soft light.');
  vm.runInContext('undoH3Enhancement()', scope);
  assert.equal(values.h3_prompt, 'A robot walks.');
  scope.api = async () => { values.h3_prompt = 'New user edits'; return {prompt: 'Stale result'}; };
  await vm.runInContext('enhanceH3Prompt()', scope);
  assert.equal(values.h3_prompt, 'New user edits');
  scope.api = async () => { throw new Error('Grok timeout'); };
  await vm.runInContext('enhanceH3Prompt()', scope);
  assert.equal(values.h3_prompt, 'New user edits');
  assert.equal(elements.h3_enhance_button.disabled, false);
  console.log('LoRA add/remove, toggles, strength persistence; Grok apply/undo, edit race, failure verified');
})().catch(error => { console.error(error); process.exitCode = 1; });
