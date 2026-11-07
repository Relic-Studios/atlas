/* Window chrome only. No application API or Node capability is exposed to pages. */
window.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('atlas-shell-title')) return;
  const top = Number.parseFloat(getComputedStyle(document.body).paddingTop) || 0;
  const style = document.createElement('style');
  style.textContent = `
    * { scrollbar-width: none !important; }
    ::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
    :root { --atlas-caption-height: env(titlebar-area-height, 32px); }
    body { padding-top: calc(${top}px + var(--atlas-caption-height)) !important; box-sizing: border-box; }
    body > .app { height: calc(100vh - var(--atlas-caption-height)) !important; }
    #atlas-shell-title { position: fixed; top: 0; left: 0; right: 140px; height: var(--atlas-caption-height);
      display: flex; align-items: center; gap: 8px; padding: 0 12px; z-index: 2147483647;
      background: #0b0d10; color: #8f98a7; font: 10px 'Segoe UI', sans-serif; letter-spacing: 2px;
      user-select: none; -webkit-app-region: drag; }
    #atlas-shell-title svg { width: 17px; height: 17px; }
  `;
  document.head.appendChild(style);
  const title = document.createElement('div');
  title.id = 'atlas-shell-title';
  title.innerHTML = '<svg viewBox="0 0 512 512" aria-hidden="true"><path d="M146 380 L256 120 L366 380" fill="none" stroke="#f3f5f7" stroke-width="52" stroke-linecap="round" stroke-linejoin="round"/><circle cx="256" cy="296" r="30" fill="#5ed3c6"/></svg><span>ATLAS</span>';
  document.body.prepend(title);
});
