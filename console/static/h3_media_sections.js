/* Collapsing a media section only changes its visibility, never its inputs. */
(() => {
  const key = 'h3-media-sections-v1';
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(key) || '{}') || {}; } catch {}
  for (const section of document.querySelectorAll('.h3MediaSection')) {
    if (typeof saved[section.id] === 'boolean') section.open = saved[section.id];
    section.addEventListener('toggle', () => {
      saved[section.id] = section.open;
      try { localStorage.setItem(key, JSON.stringify(saved)); } catch {}
    });
  }
})();
