let H3_ENHANCE_ORIGINAL = null;
function grokEnhanceContext() {
  const current = typeof h3Settings === 'function' ? h3Settings() : {task: val('h3_task'), duration: val('h3_duration')};
  const context = {use_attached_media: $('h3_enhance_media')?.checked !== false};
  const keys = context.use_attached_media ? ['task', 'duration', 'photo_paths', 'use_multi_image', 'first_frame_path', 'last_frame_path',
    'reference_paths', 'reference_video_paths', 'reference_video_settings', 'reference_video_audio', 'reference_audio_paths'] : ['task', 'duration'];
  for (const key of keys)
    if (current[key] !== undefined) context[key] = current[key];
  return context;
}
if (typeof window !== 'undefined') window.addEventListener('console-ready', () => {
  const control = $('h3_enhance_media');
  if (!control) return;
  try { control.checked = localStorage.getItem('h3-grok-use-media') !== 'false'; } catch {}
  control.onchange = () => { try { localStorage.setItem('h3-grok-use-media', String(control.checked)); } catch {} };
}, {once:true});
async function enhanceH3Prompt() {
  const button = $('h3_enhance_button'), status = $('h3_enhance_status');
  if (button.disabled) return;
  const original = val('h3_prompt');
  if (!original.trim()) { status.textContent = 'Enter a prompt first.'; return; }
  const context = grokEnhanceContext(), snapshot = JSON.stringify(context);
  button.disabled = true;
  status.textContent = context.use_attached_media ? 'Checking available attachments and enhancing with Grok…' : 'Enhancing text with Grok…';
  try {
    const result = await api('/api/h3/enhance-prompt', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: original, ...context,
        settings: {grok_api_key: val('grok_api_key'), grok_model: val('grok_model') || 'grok-4.7',
          grok_system_prompt: val('grok_system_prompt')}}),
    });
    if (val('h3_prompt') !== original || JSON.stringify(grokEnhanceContext()) !== snapshot) {
      status.textContent = 'Prompt or attachments changed while enhancing. Your edits were kept; click again to enhance them.';
      return;
    }
    H3_ENHANCE_ORIGINAL = original;
    set('h3_prompt', result.prompt);
    $('h3_enhance_undo').disabled = false;
    $('h3_prompt').dispatchEvent(new Event('input', {bubbles: true}));
    persistSettingsSoon();
    status.textContent = `Enhanced with ${result.model}. Review or edit before running H3. `
      + ((result.media_reviewed || []).length ? 'Reviewed: ' + result.media_reviewed.join('; ') + '. ' : 'No attachments reviewed. ')
      + (result.media_warnings || []).join(' ');
  } catch (error) {
    status.textContent = error.message;
  } finally { button.disabled = false; }
}
function undoH3Enhancement() {
  if (H3_ENHANCE_ORIGINAL === null) return;
  set('h3_prompt', H3_ENHANCE_ORIGINAL);
  H3_ENHANCE_ORIGINAL = null;
  $('h3_enhance_undo').disabled = true;
  $('h3_prompt').dispatchEvent(new Event('input', {bubbles: true}));
  persistSettingsSoon();
  $('h3_enhance_status').textContent = 'Original prompt restored.';
}
