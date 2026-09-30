let PB = {catalog: null, subject: '', subjectB: '', video: '', subjectFolder: '', sceneFolder: 'inspected', dirty: false, aspectTouched: false, durationTouched: false, startTouched: false, syncing: false};
const PB_PRESETS = {
  ours: {steps: 8, megapixels: 0.2, final: 1, latent: true, cache: 0.244, note: 'Ours: 0.2 MP, then a 1 MP latent refine. Split stays one below the step count.'},
  fast5090: {steps: 4, megapixels: 0.5, final: 0.5, latent: false, cache: 0.2, note: '5090 fast is one pass. Latent refine and its sigmas stay off.'},
  quality5090: {steps: 8, megapixels: 0.2, final: 1, latent: true, cache: 0.2, note: '5090 quality keeps the 1 MP refine. Set the split and second-pass sigmas in Generation.'},
};
const PB_ASPECTS = [[1, 1, '1:1 (Square)'], [2, 3, '2:3 (Portrait Photo)'], [3, 2, '3:2 (Photo)'], [3, 4, '3:4 (Portrait Standard)'], [4, 3, '4:3 (Standard)'], [9, 16, '9:16 (Portrait Widescreen)'], [16, 9, '16:9 (Widescreen)'], [21, 9, '21:9 (Ultrawide)']];

function pbEsc(value) {
  return String(value || '').replace(/[&<>"']/g, (ch) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
}

function pbChecked(id) {
  return !!$(id)?.checked;
}

function pbLoras() {
  try {
    const parsed = JSON.parse(val('pb_loras') || '[]');
    return Array.isArray(parsed) ? parsed.filter(item => item && typeof item.name === 'string') : [];
  } catch {
    return [];
  }
}

function pbRenderLoras() {
  const box = $('pb_lora_active');
  if (!box) return;
  const items = pbLoras();
  box.innerHTML = items.length
    ? items.map((item, index) => `<div class="pbLoraActiveRow"><span>${pbEsc(item.name)}</span><label>Strength<input type="number" min="0" max="2" step="0.05" value="${Number(item.strength ?? 1)}" data-pb-lora-strength="${index}"></label><button type="button" data-pb-lora-remove="${index}">Remove</button></div>`).join('')
    : '<span class="loraEmpty">No active Prompt Book LoRAs</span>';
}

async function pbLoadLoraLibrary() {
  const select = $('pb_lora_library_select');
  const status = $('pb_lora_status');
  if (!select || select.dataset.loaded) return;
  select.dataset.loaded = '1';
  try {
    const result = await api('/api/h3/loras');
    const items = Array.isArray(result.items) ? result.items : [];
    select.innerHTML = '<option value="">Choose a LoRA from the H3 volume...</option>' + items.map(item => `<option value="${pbEsc(item.name)}">${pbEsc(item.name)}${item.size_mb ? ` (${pbEsc(item.size_mb)} MB)` : ''}</option>`).join('');
    if (status) status.textContent = `${items.length} H3 LoRA${items.length === 1 ? '' : 's'} available. Uploaded/imported LoRAs appear here after refresh.`;
  } catch (error) {
    if (status) status.textContent = 'H3 LoRA library unavailable: ' + error.message;
    select.innerHTML = '<option value="">LoRA library unavailable</option>';
  }
}

function pbAddLora() {
  const name = String(val('pb_lora_library_select') || '').trim();
  const strength = Number(val('pb_lora_strength') || 1);
  if (!name) return alert('Choose an H3 LoRA first.');
  if (!Number.isFinite(strength) || strength < 0 || strength > 2) return alert('LoRA strength must be between 0 and 2.');
  const items = pbLoras();
  const existing = items.find(item => item.name === name);
  if (existing) existing.strength = strength;
  else items.push({name, strength});
  set('pb_loras', JSON.stringify(items));
  pbRenderLoras();
}

function pbRemoveLora(index) {
  const items = pbLoras();
  items.splice(index, 1);
  set('pb_loras', JSON.stringify(items));
  pbRenderLoras();
}

function loadPromptBook() {
  if (!window.PB_SAMPLING_LOADED) {
    window.PB_SAMPLING_LOADED = true;
    fillPromptBookSampling();
  }
  if (!PB.catalog) pbLoadCatalog();
  void pbLoadLoraLibrary();
  pbRenderLoras();
}

async function fillPromptBookSampling() {
  try {
    const data = await api('/api/h3/sampling');
    for (const kind of ['sampler', 'scheduler']) {
      const el = $('pb_' + kind);
      if (!el) continue;
      const current = el.value;
      for (const value of data[kind + 's'] || []) {
        if (![...el.options].some(option => option.value === value)) el.add(new Option(value.replaceAll('_', ' '), value));
      }
      if ([...el.options].some(option => option.value === current)) el.value = current;
    }
  } catch (error) {
    log('Sampler list unavailable: ' + error.message);
  }
}

async function pbLoadCatalog() {
  const status = $('promptbook_status');
  try {
    if (status) status.textContent = 'Loading prompt book...';
    PB.catalog = await api('/api/prompt-book/catalog');
    pbRenderFolders();
    pbRenderSubjects();
    pbRenderScenes();
    if (status) status.textContent = `${PB.catalog.subjects.length} subjects · ${PB.catalog.videos.length} scenes`;
  } catch (error) {
    if (status) status.textContent = error.message;
    log('Prompt book: ' + error.message);
  }
}

function pbChip(label, on, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = on ? 'on' : '';
  button.textContent = label;
  button.onclick = onClick;
  return button;
}

function pbRenderFolders() {
  const subjects = $('pb_subject_folders');
  const scenes = $('pb_scene_folders');
  if (!subjects || !scenes || !PB.catalog) return;
  subjects.innerHTML = '';
  subjects.appendChild(pbChip('All', !PB.subjectFolder, () => { PB.subjectFolder = ''; pbRenderFolders(); pbRenderSubjects(); }));
  for (const name of PB.catalog.subject_folders || []) {
    subjects.appendChild(pbChip(name, PB.subjectFolder === name, () => { PB.subjectFolder = name; pbRenderFolders(); pbRenderSubjects(); }));
  }
  scenes.innerHTML = '';
  scenes.appendChild(pbChip('Inspected', PB.sceneFolder === 'inspected', () => { PB.sceneFolder = 'inspected'; pbRenderFolders(); pbRenderScenes(); }));
  scenes.appendChild(pbChip('All', PB.sceneFolder === '', () => { PB.sceneFolder = ''; pbRenderFolders(); pbRenderScenes(); }));
  for (const name of PB.catalog.scene_folders || []) {
    scenes.appendChild(pbChip(name, PB.sceneFolder === name, () => { PB.sceneFolder = name; pbRenderFolders(); pbRenderScenes(); }));
  }
}

function pbFiltered(kind) {
  const rows = kind === 'subject' ? PB.catalog.subjects : PB.catalog.videos;
  if (kind === 'subject') return PB.subjectFolder ? rows.filter(row => row.folder === PB.subjectFolder) : rows;
  if (PB.sceneFolder === 'inspected') return rows.filter(row => row.inspected);
  if (!PB.sceneFolder) return rows;
  return rows.filter(row => row.folder === PB.sceneFolder);
}

function pbTile(row, kind) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'pbTile';
  button.dataset.id = row.id;
  if (kind === 'subject' && row.id === PB.subject) button.classList.add('on');
  if (kind === 'subject' && row.id === PB.subjectB) button.classList.add('onb');
  if (kind === 'scene' && row.id === PB.video) button.classList.add('on');
  const image = kind === 'subject'
    ? (row.thumb ? `<img src="/api/prompt-book/media/subject/${encodeURIComponent(row.thumb)}" alt="${pbEsc(row.label)}">` : `<div class="pbPh">${pbEsc(row.label)}</div>`)
    : `<img src="/api/prompt-book/media/scene/${encodeURIComponent(row.id + '.jpg')}" alt="${pbEsc(row.label)}">`;
  const meta = kind === 'subject'
    ? (row.pair ? 'pair' : (row.image_count > 1 ? `${row.image_count} photos` : ''))
    : [row.duration, row.width && row.height ? `${row.width}×${row.height}` : '', row.inspected ? '' : 'uninspected'].filter(Boolean).join(' · ');
  button.innerHTML = `${image}<div class="pbCap">${pbEsc(row.label)}${meta ? `<div class="pbMeta">${pbEsc(meta)}</div>` : ''}</div>`;
  return button;
}

function pbRenderSubjects() {
  const box = $('pb_subjects');
  if (!box || !PB.catalog) return;
  box.innerHTML = '';
  for (const row of pbFiltered('subject')) box.appendChild(pbTile(row, 'subject'));
}

function pbRenderScenes() {
  const box = $('pb_scenes');
  if (!box || !PB.catalog) return;
  box.innerHTML = '';
  for (const row of pbFiltered('scene')) box.appendChild(pbTile(row, 'scene'));
}

function pbNearestAspect(width, height) {
  const w = Number(width) || 0;
  const h = Number(height) || 0;
  if (w <= 0 || h <= 0) return '16:9 (Widescreen)';
  const ratio = w / h;
  return PB_ASPECTS.slice().sort((a, b) => Math.abs(ratio - a[0] / a[1]) - Math.abs(ratio - b[0] / b[1]))[0][2];
}

function pbPickSubject(id) {
  const three = pbChecked('pb_threesome');
  const subject = (PB.catalog?.subjects || []).find(row => row.id === id);
  if (!three || subject?.pair) {
    PB.subject = id;
    PB.subjectB = '';
  } else if (id === PB.subject) {
    PB.subject = PB.subjectB;
    PB.subjectB = '';
  } else if (id === PB.subjectB) {
    PB.subjectB = '';
  } else if (!PB.subject) {
    PB.subject = id;
  } else {
    PB.subjectB = id;
  }
  PB.dirty = false;
  pbRenderSubjects();
  pbBake(true);
}

function pbClipSeconds() {
  const video = $('pb_trim_video_preview');
  if (video && Number.isFinite(video.duration) && video.duration > 0) return video.duration;
  const scene = (PB.catalog?.videos || []).find(row => row.id === PB.video);
  return Number(scene?.duration_sec) || 0;
}

function pbApplyWindow(startSeconds, durationSeconds, clipOverride) {
  const clip = clipOverride == null ? pbClipSeconds() : clipOverride;
  let duration = Number(durationSeconds);
  if (!Number.isFinite(duration) || duration <= 0) duration = 5;
  duration = Math.min(15, Math.max(1, duration));
  let start = Number(startSeconds);
  if (!Number.isFinite(start) || start < 0) start = 0;
  if (clip > 1) start = Math.min(start, clip - 1);
  else if (clip > 0) start = 0;
  const room = clip > 0 ? Math.max(0, clip - start) : 15;
  if (room > 0) duration = Math.min(duration, room);
  duration = Math.min(15, Math.max(1, duration));
  start = Math.round(start * 10) / 10;
  duration = Math.round(duration * 10) / 10;
  PB.syncing = true;
  set('pb_start', start);
  set('pb_video_start', String(start));
  set('pb_duration', duration);
  set('pb_length', duration);
  set('pb_video_end', String(Math.round((start + duration) * 10) / 10));
  const video = $('pb_trim_video_preview');
  if (video && Number.isFinite(video.duration) && video.duration > 0) {
    const playhead = Math.min(start, Math.max(0, video.duration - 0.05));
    if (Math.abs((video.currentTime || 0) - playhead) > 0.05) video.currentTime = playhead;
  }
  if (typeof updateTimeline === 'function') updateTimeline('pb');
  PB.syncing = false;
  return duration;
}

function pbWriteDuration(seconds, clipOverride) {
  const start = Number(val('pb_start') || val('pb_video_start')) || 0;
  return pbApplyWindow(start, seconds, clipOverride);
}

function pbSyncDurationFromTrim() {
  if (PB.syncing) return;
  const start = Math.max(0, Number(val('pb_video_start')) || 0);
  const startShown = Math.round(start * 10) / 10;
  const endRaw = val('pb_video_end').trim();
  if (!endRaw) {
    if (Math.abs((Number(val('pb_start')) || 0) - startShown) < 0.05) return;
    PB.startTouched = true;
    PB.syncing = true;
    set('pb_start', startShown);
    PB.syncing = false;
    return;
  }
  const length = Number(endRaw) - start;
  if (!Number.isFinite(length) || length <= 0) return;
  const clip = pbClipSeconds();
  const room = clip > 0 ? Math.max(0, clip - start) : 15;
  let value = Math.min(15, length);
  if (room > 0) value = Math.min(value, room);
  value = Math.max(1, Math.round(value * 10) / 10);
  const startChanged = Math.abs((Number(val('pb_start')) || 0) - startShown) >= 0.05;
  const durationChanged = Math.abs((Number(val('pb_duration')) || 0) - value) >= 0.05;
  if (!startChanged && !durationChanged) return;
  if (startChanged) PB.startTouched = true;
  if (durationChanged) PB.durationTouched = true;
  PB.syncing = true;
  set('pb_start', startShown);
  set('pb_video_start', String(startShown));
  set('pb_duration', value);
  set('pb_length', value);
  PB.syncing = false;
}

function pbLoadPreview(id) {
  const video = $('pb_trim_video_preview');
  if (!video) return;
  const keep = PB.durationTouched ? Number(val('pb_duration')) : 0;
  const keepStart = PB.startTouched ? Number(val('pb_start')) : 0;
  set('pb_video_start', '');
  set('pb_video_end', '');
  video.onloadedmetadata = null;
  video.removeAttribute('src');
  const scene = (PB.catalog?.videos || []).find(row => row.id === id);
  const catalogLength = Number(scene?.duration_sec) || 0;
  pbApplyWindow(keepStart, keep || catalogLength || 5, catalogLength);
  video.onloadedmetadata = () => {
    const length = PB.durationTouched ? Number(val('pb_duration')) : video.duration;
    const start = PB.startTouched ? Number(val('pb_start')) : 0;
    pbApplyWindow(start, length, video.duration);
  };
  video.classList.remove('hidden');
  video.src = '/api/prompt-book/media/video/' + encodeURIComponent(id) + '.mp4';
  video.load();
}

function pbPickScene(id) {
  PB.video = id;
  PB.dirty = false;
  const scene = (PB.catalog?.videos || []).find(row => row.id === id);
  if (scene && !PB.aspectTouched) set('pb_aspect_ratio', pbNearestAspect(scene.width, scene.height));
  pbRenderScenes();
  const tile = document.querySelector(`#pb_scenes .pbTile[data-id="${CSS.escape(id)}"]`);
  tile?.scrollIntoView({inline: 'nearest', block: 'nearest'});
  pbLoadPreview(id);
  pbBake(true);
}

function pbPromptQuery() {
  return new URLSearchParams({
    subject: PB.subject,
    video: PB.video,
    panties: pbChecked('pb_panties') ? '1' : '0',
    sound: pbChecked('pb_sound') ? '1' : '0',
    identity_only: pbChecked('pb_identity_only') ? '1' : '0',
    bare_breasts: pbChecked('pb_bare_breasts') ? '1' : '0',
    test_mode: pbChecked('pb_test_mode') ? '1' : '0',
    video_editing: pbChecked('pb_video_editing') ? '1' : '0',
    threesome: pbChecked('pb_threesome') ? '1' : '0',
    subject_b: pbChecked('pb_threesome') ? PB.subjectB : '',
    beta: pbChecked('pb_beta') ? '1' : '0',
  });
}

async function pbBake(force) {
  const note = $('pb_prompt_note');
  if (!PB.subject || !PB.video) {
    if (note) note.textContent = 'Pick a subject and a scene. Edits here are for this send only.';
    return;
  }
  if (PB.dirty && !force) return;
  try {
    const baked = await api('/api/prompt-book/prompt?' + pbPromptQuery().toString());
    set('pb_prompt', baked.prompt || '');
    $('pb_prompt')?.classList.remove('dirty');
    PB.dirty = false;
    if (note) note.textContent = 'Edits here are for this send only. Reset reloads the baked prompt.';
  } catch (error) {
    if (note) note.textContent = error.message;
  }
}

function pbResetPrompt() {
  PB.dirty = false;
  pbBake(true);
}

function pbSyncGeneration() {
  const latent = pbChecked('pb_latent_upscale');
  for (const id of ['pb_final_megapixels', 'pb_pass1_split', 'pb_second_pass_sigma']) {
    if ($(id)) $(id).disabled = !latent;
  }
  const steps = Number(val('pb_steps')) || 2;
  const split = Number(val('pb_pass1_split')) || 1;
  if (latent && split >= steps) set('pb_pass1_split', Math.max(1, steps - 1));
  if ($('pb_cache_threshold')) $('pb_cache_threshold').disabled = !pbChecked('pb_cache_enabled');
}

function pbApplyPreset() {
  const preset = PB_PRESETS[val('pb_workflow')] || PB_PRESETS.ours;
  set('pb_steps', preset.steps);
  set('pb_megapixels', preset.megapixels);
  set('pb_final_megapixels', preset.final);
  setChk('pb_latent_upscale', preset.latent);
  setChk('pb_cache_enabled', true);
  set('pb_cache_threshold', preset.cache);
  set('pb_pass1_split', Math.max(1, preset.steps - 1));
  set('pb_second_pass_sigma', '2');
  pbSyncGeneration();
  const note = $('pb_preset_note');
  if (note) note.textContent = preset.note;
}

function pbSteps() {
  const number = Number(val('pb_steps'));
  if (!Number.isInteger(number) || number < 2 || number > 30) throw new Error('Steps must be a whole number from 2 to 30.');
  return number;
}

function pbGeneration() {
  return {
    steps: pbSteps(),
    sampler: val('pb_sampler') || 'euler',
    scheduler: val('pb_scheduler') || 'simple',
    megapixels: Number(val('pb_megapixels')),
    final_megapixels: Number(val('pb_final_megapixels')),
    latent_upscale: pbChecked('pb_latent_upscale'),
    rtx_upscale: pbChecked('pb_rtx_upscale'),
    pass1_split: Number(val('pb_pass1_split')),
    second_pass_sigma: Number(val('pb_second_pass_sigma')),
    cache_enabled: pbChecked('pb_cache_enabled'),
    cache_threshold: Number(val('pb_cache_threshold')),
    attention: val('pb_attention') || 'sage',
    aspect_ratio: val('pb_aspect_ratio'),
    seed_random: pbChecked('pb_seed_random'),
    sparse_keep_percent: Number(val('pb_sparse_keep_percent')),
    sparse_tau: Number(val('pb_sparse_tau')),
    sparse_start_percent: Number(val('pb_sparse_start_percent')),
    sparse_end_percent: Number(val('pb_sparse_end_percent')),
    sparse_trained_weights: pbChecked('pb_sparse_trained_weights'),
    loras: pbLoras(),
  };
}

async function runPromptBook() {
  const button = $('pb_run_button');
  const status = $('promptbook_status');
  try {
    if (!PB.subject || !PB.video) return alert('Pick a subject and a scene.');
    if (!val('pb_prompt').trim()) return alert('The prompt is empty.');
    if (button) {
      button.disabled = true;
      button.textContent = 'Submitting...';
    }
    if (status) status.textContent = 'Uploading the still and clip, then sending to MiniMax H3...';
    const payload = {
      h3_endpoint_id: val('h3_endpoint_id'),
      workflow_id: val('pb_workflow') || 'ours',
      subject_id: PB.subject,
      video_id: PB.video,
      subject_b: pbChecked('pb_threesome') ? PB.subjectB : '',
      use_prompt_override: true,
      prompt: val('pb_prompt'),
      panties: pbChecked('pb_panties'),
      sound: pbChecked('pb_sound'),
      identity_only: pbChecked('pb_identity_only'),
      bare_breasts: pbChecked('pb_bare_breasts'),
      test_mode: pbChecked('pb_test_mode'),
      video_editing: pbChecked('pb_video_editing'),
      threesome: pbChecked('pb_threesome'),
      beta: pbChecked('pb_beta'),
      duration: Number(val('pb_duration')) || 5,
      trim_start: val('pb_video_start') || '0',
      trim_end: val('pb_video_end') || '',
      ...pbGeneration(),
    };
    await submitRun('/api/run/prompt-book', payload, 'promptbook');
    if (status) status.textContent = 'Job submitted';
  } catch (error) {
    log('Prompt Book ERROR: ' + error.message);
    alert(error.message);
    if (status) status.textContent = 'Error';
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = 'Send to MiniMax H3';
    }
  }
}

function bindPromptBook() {
  const select = $('pb_workflow');
  if (!select || select.dataset.bound) return;
  select.dataset.bound = '1';
  select.addEventListener('change', pbApplyPreset);
  if (document.querySelector('.videoTools[data-prefix="pb"]')) {
    buildVideoTools('pb');
    mountTrimVideoPreview('pb');
    const fields = document.querySelector('#promptbook .videoTools .fields.compact');
    if (fields && !$('pb_length')) {
      const label = document.createElement('label');
      label.title = 'Generated video length, 1 to 15 seconds. This is the MiniMax H3 duration.';
      label.innerHTML = 'Length (seconds)<input id="pb_length" type="number" min="1" max="15" step="0.1" value="5">';
      fields.prepend(label);
    }
    const startInput = $('pb_video_start');
    const startLabel = startInput?.closest('label');
    if (startLabel) {
      startLabel.title = 'Where the source clip begins, in seconds.';
      if (startLabel.firstChild) startLabel.firstChild.textContent = 'Start (seconds)';
    }
    if (startInput) {
      startInput.type = 'number';
      startInput.min = '0';
      startInput.step = '0.1';
      startInput.value = val('pb_start') || '0';
      startInput.placeholder = '0';
    }
  }
  if (!PB._timelineWrapped && typeof updateTimeline === 'function') {
    const inner = updateTimeline;
    updateTimeline = function (prefix) {
      inner(prefix);
      if (prefix === 'pb') pbSyncDurationFromTrim();
    };
    PB._timelineWrapped = true;
  }
  $('pb_subjects')?.addEventListener('click', (event) => {
    const tile = event.target.closest('.pbTile');
    if (tile) pbPickSubject(tile.dataset.id);
  });
  $('pb_scenes')?.addEventListener('click', (event) => {
    const tile = event.target.closest('.pbTile');
    if (tile) pbPickScene(tile.dataset.id);
  });
  $('pb_prompt')?.addEventListener('input', () => {
    PB.dirty = true;
    $('pb_prompt').classList.add('dirty');
  });
  $('pb_reset_prompt')?.addEventListener('click', pbResetPrompt);
  $('pb_add_lora')?.addEventListener('click', pbAddLora);
  $('pb_lora_active')?.addEventListener('click', event => {
    const button = event.target.closest('[data-pb-lora-remove]');
    if (button) pbRemoveLora(Number(button.dataset.pbLoraRemove));
  });
  $('pb_lora_active')?.addEventListener('change', event => {
    const input = event.target.closest('[data-pb-lora-strength]');
    if (!input) return;
    const index = Number(input.dataset.pbLoraStrength);
    const strength = Number(input.value);
    const items = pbLoras();
    if (!Number.isInteger(index) || !items[index] || !Number.isFinite(strength) || strength < 0 || strength > 2) return;
    items[index].strength = strength;
    set('pb_loras', JSON.stringify(items));
  });
  $('pb_aspect_ratio')?.addEventListener('change', () => { PB.aspectTouched = true; });
  for (const id of ['pb_panties', 'pb_identity_only', 'pb_bare_breasts', 'pb_sound', 'pb_video_editing', 'pb_test_mode', 'pb_beta']) {
    $(id)?.addEventListener('change', () => pbBake(false));
  }
  $('pb_threesome')?.addEventListener('change', () => {
    if (!pbChecked('pb_threesome')) PB.subjectB = '';
    pbRenderSubjects();
    pbBake(false);
  });
  for (const id of ['pb_latent_upscale', 'pb_cache_enabled', 'pb_steps']) $(id)?.addEventListener('change', pbSyncGeneration);
  for (const id of ['pb_duration', 'pb_length']) {
    $(id)?.addEventListener('input', () => {
      if (PB.syncing) return;
      const raw = val(id).trim();
      if (!raw || raw.endsWith('.')) return;
      PB.durationTouched = true;
      pbWriteDuration(raw);
    });
  }
  for (const id of ['pb_start', 'pb_video_start']) {
    $(id)?.addEventListener('input', () => {
      if (PB.syncing) return;
      const raw = val(id).trim();
      if (!raw || raw.endsWith('.')) return;
      PB.startTouched = true;
      pbApplyWindow(raw, Number(val('pb_duration')) || 5);
    });
  }
  pbApplyPreset();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bindPromptBook);
else bindPromptBook();
