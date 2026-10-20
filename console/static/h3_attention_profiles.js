/* Hardware preview never pins a GPU: the allocated worker is authoritative. */
(() => {
  let catalog;
  const value = id => document.getElementById(id)?.value || '';
  const selected = prefix => value(prefix + '_attention_profile') || 'custom';
  const wrap = (name, prefix) => {
    const original = window[name];
    window[name] = (...args) => ({...original(...args), attention_profile: selected(prefix)});
  };
  wrap('h3Settings', 'h3');
  wrap('pbGenerationSettings', 'pb');
  const applySaved = window.applyH3Settings;
  window.applyH3Settings = (settings, ...args) => {
    const result = applySaved(settings, ...args);
    const control = document.getElementById('h3_attention_profile');
    if (control) control.value = settings?.attention_profile || 'custom';
    return result;
  };
  function refresh(prefix) {
    const id = selected(prefix), profile = catalog.profiles[id];
    const card = catalog.cards[value(prefix + '_attention_card')];
    const hint = document.getElementById(prefix + '_attention_profile_hint');
    hint.textContent = [card ? card.label + ': ' + card.encoder + ' encoder, Sage dense default.' : catalog.note,
                        profile?.evidence || ''].join(' ');
    const sla = document.querySelector(`#${prefix}_attention_profile option[value="sla"]`);
    if (sla) {
      const settings = prefix === 'h3' ? h3Settings() : {};
      sla.disabled = prefix !== 'h3' || !['fl2v', 't2v'].includes(settings.task)
        || settings.fl2va_model !== catalog.profiles.sla.required_model
        || !(settings.loras || []).some(l => String(l.name || '').split(/[\\/]/).pop() === catalog.profiles.sla.required_adapter);
    }
  }
  function install(prefix, saved = 'custom') {
    const attention = document.getElementById(prefix + '_attention');
    if (!attention || document.getElementById(prefix + '_attention_profile')) return;
    const container = document.createElement('div'); container.className = 'card attentionProfiles';
    container.innerHTML = `<h4>Attention workflow</h4><label>Card preview<select id="${prefix}_attention_card"><option value="">Allocated worker · automatic</option></select></label><label>Configuration<select id="${prefix}_attention_profile"></select></label><p id="${prefix}_attention_profile_hint" class="hint" role="status"></p>`;
    attention.closest('label').after(container);
    const cards = document.getElementById(prefix + '_attention_card');
    for (const [id, item] of Object.entries(catalog.cards)) {
      const option = document.createElement('option'); option.value = id; option.textContent = item.label; cards.append(option);
    }
    const profiles = document.getElementById(prefix + '_attention_profile');
    for (const [id, item] of Object.entries(catalog.profiles)) {
      const option = document.createElement('option'); option.value = id; option.textContent = item.label; option.disabled = !!item.disabled; profiles.append(option);
    }
    profiles.value = catalog.profiles[saved] && !catalog.profiles[saved].disabled ? saved : 'custom';
    cards.onchange = () => refresh(prefix);
    profiles.onchange = () => {
      const item = catalog.profiles[profiles.value];
      if (item.attention) {
        attention.value = item.attention;
        const cache = document.getElementById(prefix + '_cache_enabled'); if (cache) cache.checked = false;
      }
      refresh(prefix);
      if (prefix === 'h3') window.persistSettingsSoon?.();
    };
    attention.addEventListener('change', () => { profiles.value = 'custom'; refresh(prefix); });
    profiles.addEventListener('focus', () => refresh(prefix));
    refresh(prefix);
  }
  window.addEventListener('console-ready', async () => {
    try {
      catalog = await api('/api/h3/attention-profiles');
      install('h3', SETTINGS.h3?.attention_profile);
      let saved = {}; try { saved = JSON.parse(localStorage.getItem('prompt-book-h3-generation-v1') || '{}'); } catch {}
      install('pb', saved.attention_profile);
    } catch (error) { log('Attention configurations unavailable: ' + error.message); }
  }, {once: true});
})();
