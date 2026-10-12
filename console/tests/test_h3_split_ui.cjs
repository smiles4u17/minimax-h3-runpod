const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const source = fs.readFileSync('static/h3_variants.js', 'utf8');
const values = {h3_steps: '8', h3_pass1_split: '6', h3_second_pass_sigma: '3', h3_sampler: 'euler'};
const checked = {h3_no_turbo: true, h3_latent_upscale: true};
const elements = {};
const node = () => ({classList: {toggle(){}, add(){}}, closest(){return this;}});
const get = id => elements[id] ||= node();
get('h3_reference_grid').previousElementSibling = node();
get('h3_reference_grid').previousElementSibling.previousElementSibling = node();
let saves = 0;
const ctx = {$: get, val: id => values[id] || '', chk: id => !!checked[id],
  set: (id, value) => values[id] = String(value), setChk: (id, value) => checked[id] = value,
  isNew: () => true, variant: () => 'fflf_20260920', photos: node(),
  document: {querySelector: () => node()}, updateH3TaskUI(){}, updateH3TurboUI(){},
  persistSettingsSoon(){saves++;}};
vm.createContext(ctx);
vm.runInContext(source.slice(source.indexOf('  function sync('), source.indexOf("  $('h3_task').addEventListener")), ctx);
vm.runInContext('sync()', ctx);
assert.equal(get('h3_pass1_split').disabled, false, 'No Turbo split must stay editable');
assert.match(get('h3_variant_notice').textContent, /complete the first pass/);
vm.runInContext(source.split('\n').find(line => line.includes("$('h3_pass1_split').onchange=")), ctx);
values.h3_pass1_split = '5';
get('h3_pass1_split').onchange();
assert.equal(values.h3_second_pass_sigma, '4', 'Editing split selects a schedule that honors it');
assert.equal(values.h3_pass1_split, '5');
assert.equal(saves, 1);
assert.equal(get('h3_pass1_split').max, 7);
checked.h3_latent_upscale = false;
vm.runInContext('sync()', ctx);
assert.equal(get('h3_pass1_split').disabled, true);
assert.match(source, /value="5">5 — 8 refine steps/);
console.log('H3 editable split selects remaining sigmas; latent-off disabling passed');
