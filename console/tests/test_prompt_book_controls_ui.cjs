const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const catalog = JSON.parse(fs.readFileSync(path.join(root, 'h3_sampling.json'), 'utf8'));
const source = fs.readFileSync(path.join(root, 'static', 'prompt_book.js'), 'utf8');

function select(initial) {
  let current = initial;
  return {
    options: [{value: initial}],
    get value() { return current; },
    set value(value) { current = value; },
    set innerHTML(html) {
      this.options = [...html.matchAll(/<option value="([^"]*)"/g)].map(match => ({value: match[1]}));
      current = this.options[0]?.value || '';
    },
    add(option) { this.options.push(option); },
  };
}

const elements = {
  pb_sampler: select('euler'), pb_scheduler: select('simple'),
  pb_model: select('10Eros_Max_h3_TURBO-hybrid_beta5.safetensors'),
  pb_h3_settings_summary: {textContent: ''},
};
const values = {
  pb_steps: '20', pb_pass1_split: '14', pb_second_pass_sigma: '5',
  pb_megapixels: '0.2', pb_final_megapixels: '1.0',
  pb_seed: '42', pb_attention: 'sage', pb_cache_threshold: '0.18',
  pb_loras: '[]',
};
const checked = {pb_latent_upscale: true, pb_seed_random: false};
const context = {
  document: {readyState: 'loading', addEventListener() {}},
  $: id => elements[id] || (id in checked ? {checked: checked[id]} : undefined),
  val: id => values[id] ?? elements[id]?.value ?? '',
  set: (id, value) => { values[id] = String(value); },
  payloadSettings: () => ({}),
  alert: message => { throw new Error(message); },
  api: async route => route === '/api/h3/sampling'
    ? catalog
    : {categories: {ref2va: ['selected-ref2v.safetensors'], diffusion_models: ['wrong-fl2v.safetensors']}},
  Option: function Option(label, value) { this.label = label; this.value = value; },
  log: message => { throw new Error(message); },
  h3Settings: () => { throw new Error('Prompt Book must not read the H3 tab'); },
};
vm.createContext(context);
vm.runInContext(source, context);

(async () => {
  await vm.runInContext('pbLoadGenerationOptions()', context);
  assert.deepEqual(elements.pb_sampler.options.map(option => option.value),
    catalog.samplers.filter(name => name !== 'h3_turbo'));
  assert.deepEqual(elements.pb_scheduler.options.map(option => option.value), catalog.schedulers);
  assert.deepEqual(elements.pb_model.options.map(option => option.value),
    ['10Eros_Max_h3_TURBO-hybrid_beta5.safetensors', 'selected-ref2v.safetensors']);
  elements.pb_sampler.value = 'res_multistep';
  elements.pb_scheduler.value = 'karras';
  const generation = vm.runInContext('pbGenerationSettings()', context);
  assert.equal(generation.sampler, 'res_multistep');
  assert.equal(generation.scheduler, 'karras');
  assert.equal(generation.steps, 20);
  assert.equal(generation.pass1_split, 20, 'Manual refine finishes the first pass');
  assert.equal(generation.second_pass_sigma, 5);
  assert.equal(generation.seed, 42);
  values.pb_second_pass_sigma = '4';
  assert.equal(vm.runInContext('pbGenerationSettings()', context).pass1_split, 14,
    'Explicit remaining sigmas retain the user split');
  values.pb_second_pass_sigma = '5';
  assert.equal(elements.pb_h3_settings_summary.textContent.includes('separate Turbo off'), true);
  Object.assign(values, {pb_duration: '15', pb_video_start: '145', pb_video_end: '147',
    pb_start: '145', pb_prompt: 'A neutral reference test.'});
  vm.runInContext("PB.subject = 'neutral'; PB.video = 'neutral'; PB.catalog = {videos: [{id: 'neutral', duration_sec: 200}]};", context);
  let sent;
  context.submitRun = async (route, payload) => { sent = payload; };
  await vm.runInContext('runPromptBook()', context);
  assert.equal(sent.duration, 2, 'Submission must read the visible trim rather than stale duration');
  assert.equal(sent.trim_start, '145');
  assert.equal(sent.trim_end, '147');
  values.pb_video_end = '145';
  sent = null;
  await assert.rejects(vm.runInContext('runPromptBook()', context), /Trim end must be after/);
  assert.equal(sent, null);
  process.stdout.write(`Prompt Book UI: ${elements.pb_sampler.options.length} samplers, ${elements.pb_scheduler.options.length} schedulers; independent controls OK\n`);
})().catch(error => { console.error(error); process.exitCode = 1; });
