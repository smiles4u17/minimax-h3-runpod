/* Explicit presets. Loading this file never changes the user's LCM/custom setup. */
(() => {
  const snapshots = {};
  try { Object.assign(snapshots, JSON.parse(localStorage.getItem('h3-acceleration-custom-settings-v1') || '{}')); } catch {}
  const saveSnapshots = () => { try { localStorage.setItem('h3-acceleration-custom-settings-v1', JSON.stringify(snapshots)); } catch {} };
  const mode = prefix => val(prefix + '_acceleration_recipe') || 'custom';
  const wrap = (name, prefix) => {
    const original = window[name];
    window[name] = (...args) => ({...original(...args), acceleration_recipe: mode(prefix)});
  };
  wrap('h3Settings', 'h3'); wrap('pbGenerationSettings', 'pb');
  const applySaved = window.applyH3Settings;
  window.applyH3Settings = (settings, ...args) => {
    const result = applySaved(settings, ...args);
    set('h3_acceleration_recipe', settings?.acceleration_recipe || 'custom');
    return result;
  };
  function fields(prefix) {
    const names = ['sampler','scheduler','steps','pass1_split','second_pass_sigma','attention',
      'attention_profile','cache_enabled','latent_upscale','megapixels'];
    if (prefix === 'h3') names.push('no_turbo','use_larry','sparse_trained_weights',
      'sparse_keep_percent','sparse_start_percent','sparse_end_percent',
      'fl2va_model','ref2va_model','loras');
    else names.push('model','loras');
    return names.map(name => prefix + '_' + name).filter(id => $(id));
  }
  function persist(prefix) {
    if (prefix === 'h3') persistSettingsSoon(); else pbSaveGeneration();
  }
  async function choose(prefix) {
    const selected = mode(prefix), status = $(prefix + '_acceleration_status');
    const task = prefix === 'pb' ? 'r2v' : val('h3_task');
    if (selected === 'sla4' && task === 'r2v') {
      set(prefix + '_acceleration_recipe', 'custom');
      status.textContent = 'SLA weights are FL2V/T2V only. Use PDD or Sol for Ref2V.';
      return;
    }
    if (selected === 'custom') {
      for (const [id, value] of Object.entries(snapshots[prefix] || {})) {
        if ($(id)?.type === 'checkbox') setChk(id, value); else set(id, value);
      }
      delete snapshots[prefix];
      saveSnapshots();
      if (prefix === 'h3') {
        $('h3_no_turbo')?.dispatchEvent(new Event('change', {bubbles: true}));
        renderH3LoraActive();
      }
      status.textContent = 'Custom settings restored. No acceleration preset is applied.';
      persist(prefix); return;
    }
    if (!snapshots[prefix]) snapshots[prefix] = Object.fromEntries(fields(prefix).map(id =>
      [id, $(id).type === 'checkbox' ? chk(id) : val(id)]));
    saveSnapshots();
    // Switch recipes from the saved custom settings, avoiding stacked adapters.
    for (const [id, value] of Object.entries(snapshots[prefix])) {
      if ($(id)?.type === 'checkbox') setChk(id, value); else set(id, value);
    }
    const pdd = selected === 'pdd8', steps = pdd ? 8 : 4;
    const model = task === 'r2v' ? 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'
      : 'minimax_h3_fl2va_pruned_int8_convrot.safetensors';
    const modelControl = prefix === 'pb' ? 'pb_model' : task === 'r2v' ? 'h3_ref2va_model' : 'h3_fl2va_model';
    const select = $(modelControl);
    if (select && !Array.from(select.options).some(option => option.value === model)) select.add(new Option(model, model));
    set(modelControl, model);
    set(prefix + '_sampler', 'euler'); set(prefix + '_scheduler', 'simple');
    set(prefix + '_steps', steps); set(prefix + '_pass1_split', steps);
    if (val(prefix + '_second_pass_sigma') === '4') set(prefix + '_second_pass_sigma', '2');
    setChk(prefix + '_cache_enabled', false);
    set(prefix + '_attention', pdd ? 'auto' : 'sla');
    set(prefix + '_attention_profile', pdd ? 'auto' : 'sla');
    if (prefix === 'h3') {
      setChk('h3_no_turbo', true); setChk('h3_use_larry', false);
      setChk('h3_sparse_trained_weights', !pdd);
      if (!pdd) {
        set('h3_sparse_keep_percent', 15); set('h3_sparse_start_percent', 0); set('h3_sparse_end_percent', 1);
        const items = h3Loras();
        const name = 'H3/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors';
        const existing = items.find(item => item.name.split('/').pop() === name.split('/').pop());
        if (existing) { existing.strength = 1; existing.enabled = true; }
        else items.push({name, strength: 1, enabled: true});
        setH3Loras(items, false);
      }
      $('h3_no_turbo').dispatchEvent(new Event('change', {bubbles: true}));
    }
    if (prefix === 'pb') pbUpdateGenerationControls();
    status.textContent = pdd
      ? 'PDD uses its full output heads and exact 8-step Euler schedule. Refinement uses the dense base model. Requires the updated worker image.'
      : 'SLA uses the paired FL2VA model/LoRA, 15% exact blocks, Euler 4 steps, and 6/3 shifts. No Larry sampler.';
    persist(prefix);
  }
  function install(prefix, saved) {
    const anchor = $(prefix + '_sampler');
    if (!anchor || $(prefix + '_acceleration_recipe')) return;
    const box = document.createElement('div'); box.className = 'card';
    box.innerHTML = `<h4>Acceleration recipe</h4><label>Recipe<select id="${prefix}_acceleration_recipe"><option value="custom">Custom — keep current sampler/model</option><option value="pdd8">PDD — full head bank, Euler 8 steps</option><option value="sla4" ${prefix === 'pb' ? 'disabled' : ''}>SLA — paired FL2V weights, Euler 4 steps</option></select></label><p id="${prefix}_acceleration_status" class="hint" role="status">Presets apply only when selected. Sol is available in Attention workflow. VSA requires compatible FastH3 weights and is unavailable.</p>`;
    anchor.closest('label').parentElement.after(box);
    set(prefix + '_acceleration_recipe', saved || 'custom');
    $(prefix + '_acceleration_recipe').onchange = () => void choose(prefix);
  }
  window.addEventListener('console-ready', () => {
    install('h3', SETTINGS.h3?.acceleration_recipe);
    let saved = {}; try { saved = JSON.parse(localStorage.getItem('prompt-book-h3-generation-v1') || '{}'); } catch {}
    install('pb', saved.acceleration_recipe);
  }, {once: true});
})();
