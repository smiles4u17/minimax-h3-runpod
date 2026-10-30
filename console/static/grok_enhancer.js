let H3_ENHANCE_ORIGINAL = null;
async function enhanceH3Prompt() {
  const button = $('h3_enhance_button'), status = $('h3_enhance_status');
  if (button.disabled) return;
  const original = val('h3_prompt');
  if (!original.trim()) { status.textContent = 'Enter a prompt first.'; return; }
  button.disabled = true;
  status.textContent = 'Enhancing with Grok…';
  try {
    const result = await api('/api/h3/enhance-prompt', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: original, task: val('h3_task'), duration: val('h3_duration'),
        settings: {grok_api_key: val('grok_api_key'), grok_model: val('grok_model') || 'grok-4.7',
          grok_system_prompt: val('grok_system_prompt')}}),
    });
    if (val('h3_prompt') !== original) {
      status.textContent = 'Prompt changed while enhancing. Your edits were kept; click again to enhance them.';
      return;
    }
    H3_ENHANCE_ORIGINAL = original;
    set('h3_prompt', result.prompt);
    $('h3_enhance_undo').disabled = false;
    $('h3_prompt').dispatchEvent(new Event('input', {bubbles: true}));
    persistSettingsSoon();
    status.textContent = `Enhanced with ${result.model}. Review or edit before running H3.`;
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
