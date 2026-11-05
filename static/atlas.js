// First run on a public build: send the user to the setup wizard.
fetch('/api/setup/status').then((r) => r.json()).then((s) => { if (s.setup_needed) location.replace('/static/setup.html'); }).catch(() => {});

/* ATLAS live dashboard.
 *
 * Two sockets:
 *   /ws         control + streaming chat (persona, voice, devices, partial/final text,
 *               browser-mode audio). Opened once; auto-reconnects.
 *   /telemetry  read-only feed: `fast` (~15 Hz meters/spectrum/state), `snap` (~1 Hz
 *               GPU/models/pipeline/room/threads), `events`, `lines`. A dashboard
 *               connection never touches the realtime path.
 * Everything animated runs in one requestAnimationFrame loop off smoothed state.
 */
'use strict';
const $ = (id) => document.getElementById(id);
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const hhmm = (t) => { const d = t ? new Date(t * 1000) : new Date(); return d.toTimeString().slice(0, 8); };

const S = {
  personas: [],  // filled from the server's agent registry on connect
  voices: [], persona: '', voice: '', switching: null, switchSeq: 0, voicePending: null,
  chat: [], typingUser: '', typingUserSpk: '', typingAssistant: '',
  fast: { in: 0, clean: 0, out: 0, spec: [], speaking: false, recording: false, thinking: false, partial: '' },
  snap: null, bridge: false, wsOpen: false, telOpen: false,
  sm: { in: 0, clean: 0, out: 0, spec: new Array(32).fill(0), energy: 0, think: 0 },
  scope: [], turnEndAt: 0, lastDecision: null, lastDecisionAt: 0,
  evTab: 'events',
};

/* ---------------------------------------------------------------- theme */
function hexRgb(h) { const m = h.replace('#', ''); const n = parseInt(m.length === 3 ? m.split('').map((c) => c + c).join('') : m, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
let accentRGB = hexRgb('#5ed3c6'), targetRGB = accentRGB.slice();
function persona(id) { return S.personas.find((p) => p.id === id) || { id, name: id, accent: '#5ed3c6', role: '', voice: '' }; }
function applyTheme() { targetRGB = hexRgb(persona(S.persona).accent); $('orbName').textContent = persona(S.persona).name.toUpperCase(); }
let _themeKey = '';
function tickTheme() {
  accentRGB = accentRGB.map((v, i) => lerp(v, targetRGB[i], 0.12));
  const r = accentRGB.map(Math.round), k = r.join(',');
  if (k === _themeKey) return false;     // converged: no style writes, no restyle of the page
  _themeKey = k;
  document.documentElement.style.setProperty('--accent', `rgb(${k})`);
  document.documentElement.style.setProperty('--accent-rgb', k);
  return true;
}
const rgba = (a) => `rgba(${accentRGB.map(Math.round).join(',')},${a})`;
const SPK_COLORS = ['#5eead4', '#fbbf24', '#f472b6', '#60a5fa', '#a3e635', '#fb7185', '#c084fc', '#38bdf8'];
function spkColor(label) { if (!label || label === 'you' || label === 'You') return '#5eead4'; let h = 0; for (const c of label) h = (h * 31 + c.charCodeAt(0)) >>> 0; return SPK_COLORS[h % SPK_COLORS.length]; }

/* ---------------------------------------------------------------- toast */
let toastT = 0;
function toast(msg, color) { const t = $('toast'); t.textContent = msg; t.style.borderColor = color || rgba(.6); t.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 2600); }

/* ---------------------------------------------------------------- persona UI: agent dock */
const initial = (p) => (p.name || p.id || '?').trim().charAt(0).toUpperCase();
let dockSig = '';
function renderVoiceSelect() {
  const vs = $('voiceSel'); if (!vs) return;
  const list = S.voices.length ? S.voices : [S.voice];
  const sig = list.join(',') + '|' + S.voice;
  if (vs.dataset.sig !== sig) {
    vs.innerHTML = ''; vs.dataset.sig = sig;
    list.forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; o.selected = v === S.voice; vs.appendChild(o); });
  }
  vs.disabled = !!(S.voicePending || S.switching);   // answered when the new voice has warmed up
  vs.onchange = async () => {
    const v = vs.value; S.voicePending = v; vs.disabled = true;
    const { status, data } = await postAction('/api/active_voice', { voice: v });
    S.voicePending = null;
    if (data && data.current) applyCurrent(data.current); else renderVoiceSelect();
    if (status === 200) toast(`voice → ${S.voice}`);
    else { vs.value = S.voice; toast(status === 0 ? "the backend didn't answer" : (data.error || `voice switch failed (${status})`), '#f87171'); }
  };
}
function renderDock() {
  const box = $('dock'); if (!box) return;
  const sig = S.personas.map((p) => `${p.id}:${p.accent}:${p.name}`).join('|');
  if (sig !== dockSig) {
    dockSig = sig; box.innerHTML = '';
    S.personas.forEach((p) => {
      const d = document.createElement('div');
      d.className = 'av'; d.dataset.id = p.id; d.tabIndex = 0;
      d.style.setProperty('--pc', p.accent); d.textContent = initial(p);
      d.setAttribute('aria-label', p.name);
      if (p.custom) { const c = document.createElement('i'); c.className = 'cu'; d.appendChild(c); }
      d.onclick = () => avClick(p);
      d.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); avClick(p); } };
      d.onmouseenter = () => showTip(d, p); d.onmouseleave = hideTip; d.onfocus = () => showTip(d, p); d.onblur = hideTip;
      d.oncontextmenu = (e) => { e.preventDefault(); hideTip(); openMemory(p); };
      box.appendChild(d);
    });
    const add = document.createElement('div');
    add.className = 'av add'; add.textContent = '+'; add.tabIndex = 0; add.setAttribute('aria-label', 'Create agent');
    add.onclick = openCreator; add.onkeydown = (e) => { if (e.key === 'Enter') openCreator(); };
    add.onmouseenter = () => showTip(add, { name: 'New agent', role: 'create one from a description', voice: '' }); add.onmouseleave = hideTip;
    box.appendChild(add);
  }
  for (const el of box.querySelectorAll('.av[data-id]')) {
    const id = el.dataset.id;
    el.classList.toggle('active', id === S.persona);
    el.classList.toggle('switching', S.switching === id);
    el.classList.toggle('speaking', id === S.persona && !!S.fast.speaking);
    el.classList.toggle('bg', isBackground(id));
  }
}
// Background mode (owner 10-06): click the ACTIVE agent to toggle it. The agent then
// only answers when spoken to; its name opens a short direct exchange.
const isBackground = (id) => ((S.snap?.owner?.background) || []).includes(id);
async function avClick(p) {
  if (S.switching) { if (p.id !== S.persona) setPersona(p.id); return; }   // the active chip waits while switching
  if (p.id !== S.persona) return setPersona(p.id);
  const on = !isBackground(p.id);
  const st = await ownerPost({ background: { agent: p.id, on } });
  if (!st) return toast('could not change background mode');
  renderDock(); hideTip();
  toast(on ? `${p.name || p.id} · background mode: answers only when spoken to`
           : `${p.name || p.id} · conversation mode`, p.accent);
}
function showTip(el, p) {
  const t = $('avTip'); const r = el.getBoundingClientRect();
  const bgm = p.id && isBackground(p.id);
  const st = p.id ? (S.switching === p.id ? 'switching…' : p.id === S.persona ? (S.fast.speaking ? 'speaking' : (bgm ? 'background mode · click for conversation' : 'active · click for background mode')) : (bgm ? 'click to switch · background' : 'click to switch')) : '';
  t.innerHTML = `<div class="n" style="color:${esc(p.accent || 'var(--accent)')}">${esc(p.name || '')}</div>`
    + `<div class="r">${esc(p.role || '')}</div>`
    + (p.voice || st ? `<div class="m">${p.voice ? 'voice · ' + esc(p.voice) : ''}${p.voice && st ? ' — ' : ''}${esc(st)}${p.id ? ' · right-click: memory' : ''}</div>` : '');
  t.style.left = Math.max(8, Math.min(innerWidth - 250, r.left + r.width / 2 - 80)) + 'px';
  t.style.top = (r.bottom + 10) + 'px';
  t.classList.add('show');
}
function hideTip() { $('avTip').classList.remove('show'); }
// Old call sites keep working.
function renderPersonaSelects() { renderVoiceSelect(); renderDock(); }
function renderPersonaCards() { renderDock(); }
// Agent switch: the clicked chip shows a ring until the server answers that request; the active
// marker always shows the server's agent. The server runs the switch off its event loop (it stops the running reply,
// then warms up the new voice) and answers the request itself, so each response belongs to exactly one click.
function applyCurrent(cur) {
  if (!cur || !cur.persona) return;
  const changed = cur.persona !== S.persona;
  S.persona = cur.persona; if (cur.voice) S.voice = cur.voice;
  if (changed) applyTheme();
  renderPersonaSelects();
}
async function postAction(url, body) {
  // -> {status, data}; status 0 = no answer (network error, server stopped, or a body that isn't JSON)
  try {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
    let data = null; try { data = await r.json(); } catch (_) { return { status: 0, data: null }; }
    return { status: r.status, data: data || {} };
  } catch (_) { return { status: 0, data: null }; }
}
async function setPersona(id) {
  if (!id || id === S.switching || (id === S.persona && !S.switching)) return renderDock();
  hideTip();
  const seq = ++S.switchSeq;
  S.switching = id; renderPersonaSelects();
  clearTimeout(switchTimer);
  switchTimer = setTimeout(() => {
    if (seq !== S.switchSeq || !S.switching) return;
    endSwitch(); toast(`still switching to ${persona(id).name} on the server`, '#f87171');
  }, SWITCH_TIMEOUT_MS);
  const { status, data } = await postAction('/api/active_agent', { persona: id });
  if (data && data.current) applyCurrent(data.current);    // the marker always follows the server
  if (seq !== S.switchSeq || S.switching !== id) return;    // a newer click owns the ring now
  if (status === 200) confirmSwitch(id);
  else if (status === 409) endSwitch();                     // replaced by a newer switch: no toast
  else {
    // the server's own reason, unless it's the generic one (which names the agent by its id)
    const why = data && data.error && !data.error.startsWith("couldn't switch") ? `: ${data.error}` : '';
    endSwitch(); toast(status === 0 ? "the backend didn't answer" : `couldn't switch to ${persona(id).name}${why}`, '#f87171');
  }
}
function confirmSwitch(id) {
  endSwitch();
  S.chat.push({ role: 'sys', content: `— switched to ${persona(S.persona).name} —` }); renderChat();
  toast(`agent → ${persona(S.persona).name}`, persona(S.persona).accent);
}
async function refreshPersonas() {
  try {
    const r = await fetch('/api/personas'); const m = await r.json();
    applyPersonasMsg(m);
  } catch (e) { /* ws list_personas will cover it */ }
}
// one switch already running and this one, each up to a ~15 s reply abort plus the voice warm-up (design D2)
const SWITCH_TIMEOUT_MS = 40000;
let switchTimer = 0;
function endSwitch() { clearTimeout(switchTimer); S.switching = null; renderPersonaSelects(); }
// Agent and voice lists from the server (the /ws "personas" message, /api/personas). They set the lists and the active
// marker; they never show toasts (the REST responses report this dashboard's own actions).
function applyPersonasMsg(m) {
  if (m.personas) S.personas = m.personas.map((p) => ({ id: p.id, name: p.name, role: p.role, accent: p.accent, voice: p.voice, custom: !!p.custom }));
  if (m.voices) S.voices = m.voices;
  if (m.current) applyCurrent(m.current); else renderPersonaSelects();
}
async function deleteAgent(p) {
  if (p.id === S.persona) { toast('switch to another agent before deleting this one', '#f87171'); return; }
  if (!confirm(`Delete agent ${p.name}? Its prompt file is kept as ${p.id}.deleted.`)) return;
  const r = await fetch(`/api/agents/${encodeURIComponent(p.id)}`, { method: 'DELETE' });
  const m = await r.json();
  if (!r.ok) { toast(m.error || 'delete failed', '#f87171'); return; }
  applyPersonasMsg(m); toast(`deleted ${p.name}`);
}

/* ---------------------------------------------------------------- agent memory panel */
// Owner 10-03: every agent gets a delete button with a time window, clearing ONLY that
// agent's memories, so a crowded/broken memory is always something the user can fix.
const MEM = { p: null, armT: 0 };
const MEM_LABEL = { '1h': 'the past hour', '12h': 'the past 12 hours', '1d': 'the past day', '1w': 'the past week', all: 'everything' };
function memHint() {
  const w = $('memWindow').value, n = MEM.p ? MEM.p.name : '';
  $('memHint').textContent = w === 'all'
    ? `Deletes everything ${n} remembers, including memories you approved. Other agents keep theirs. Can't be undone.`
    : `Deletes what ${n} learned or saved in ${MEM_LABEL[w]}. Older memories and other agents are untouched. Can't be undone.`;
  disarmWipe();
}
function disarmWipe() { clearTimeout(MEM.armT); const b = $('memWipe'); b.classList.remove('arm'); b.textContent = 'Delete memories'; }
async function loadMemCounts() {
  const box = $('memCounts'); const p = MEM.p; if (!p) return;
  try {
    const r = await fetch(`/api/memory/${encodeURIComponent(p.id)}`); const m = await r.json();
    if (!r.ok) throw new Error(m.error || r.status);
    if (MEM.p !== p) return;
    const eps = m.episodes || [], rev = m.review || [];
    box.innerHTML = `<span><b>${eps.length}</b> call${eps.length === 1 ? '' : 's'} remembered</span>`
      + (rev.length ? `<span><b>${rev.length}</b> to review</span>` : '');
    $('memEpisodes').innerHTML = eps.length ? eps.map((e) => `<div class="memitem"><span class="tx"><i>${esc(e.date || '')}</i>${esc(e.summary || '')}</span>`
      + `<button data-ep="${esc(e.id)}" title="Delete this memory">×</button></div>`).join('')
      : '<div class="hint">No calls remembered yet.</div>';
    $('memReview').innerHTML = rev.map((r) => `<div class="memitem"><span class="tx"><i>${r.kind === 'lore' ? 'lore' : 'lesson'}</i>${esc(r.text)}`
      + `<span class="tag">names a person?</span></span><button data-rv="${esc(r.kind)}|${esc(r.id)}|approve" title="Keep it">✓</button>`
      + `<button data-rv="${esc(r.kind)}|${esc(r.id)}|delete" title="Delete it">×</button></div>`).join('');
  } catch (e) { box.innerHTML = `<span>memory unavailable: ${esc(String(e.message || e))}</span>`; }
  loadMemShared();
}
// The experience memory's shared part (docs/designs/experience_memory.md, D7): facts about the owner, the switches,
// waiting calls, the allow-list and the pass log.
async function memPost(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  let m = {}; try { m = await r.json(); } catch (_) {}
  if (!r.ok) throw new Error(m.error || r.status);
  return m;
}
async function loadMemShared() {
  try {
    const r = await fetch('/api/memory_state'); const s = await r.json();
    if (!r.ok) throw new Error(s.error || r.status);
    $('memUse').checked = !!s.settings.use_memories;
    $('memUse').disabled = !s.settings.use_memories && !(s.report && s.report.passed);
    $('memPause').checked = !!s.settings.paused;
    $('memCloud').checked = !!s.settings.cloud_allowed;
    const rep = s.report || {};
    $('memReport').textContent = !rep.exists ? 'Agents use only what you ask them to remember until the memory checks pass.'
      : rep.passed ? `Checks passed ${rep.date || ''}${rep.shipped ? ' (measured on the developer\'s setup, not your model)' : ''}.`
      : rep.code_ok === false ? 'The last check was for an older version of the memory; it needs to run again.'
      : 'The last memory check did not pass, so agents use only what you ask them to remember.';
    const facts = s.facts || [];
    $('memFacts').innerHTML = facts.length ? facts.map((f) => `<div class="memitem"><span class="tx"><i>${esc(f.date || '')}</i>${esc(f.fact || '')}`
      + (f.confirmed ? '' : '<span class="tag">to confirm</span>') + `</span>`
      + (f.confirmed ? '' : `<button data-fc="${esc(f.id)}" title="Yes, keep it">✓</button>`)
      + `<button data-fd="${esc(f.id)}" title="Delete">×</button></div>`).join('')
      : '<div class="hint">Nothing yet. Say "remember that ..." in a call.</div>';
    const w = (s.waiting || []).length, fl = (s.failed || []).length;
    $('memCalls').textContent = (s.call_in_progress ? 'A call is in progress. ' : '')
      + `${w} call${w === 1 ? '' : 's'} waiting`
      + (w ? ` (oldest ${Math.max(...s.waiting.map((c) => c.age_h || 0))} h)` : '') + (fl ? `, ${fl} failed` : '')
      + (s.tagger && !s.tagger.present ? (s.tagger.downloading ? '. Downloading the name tagger…' : '. The name tagger is missing.') : '')
      + (s.runner && s.runner.running ? '. Making memories now…' : '');
    $('memProcess').textContent = s.tagger && !s.tagger.present ? 'Download tagger' : 'Process now';
    $('memProcess').dataset.tagger = s.tagger && !s.tagger.present ? '1' : '';
    $('memAllow').value = (s.allow || []).join('\n');
    $('memOld').textContent = s.old_data ? `Delete old memory data (${s.old_data})` : 'No old memory data';
    $('memOld').disabled = !s.old_data;
  } catch (e) { $('memReport').textContent = `memory status unavailable: ${e.message || e}`; }
}
async function loadMemLog() {
  try { const r = await fetch('/api/memory_state/log'); const s = await r.json(); $('memLog').textContent = (s.lines || []).slice(-60).join('\n') || 'No runs yet.'; }
  catch (_) { $('memLog').textContent = ''; }
}
function armTwice(btn, label, fn) {                 // destructive buttons: first click arms, second does it
  if (!btn.classList.contains('arm')) {
    btn.dataset.label = btn.textContent; btn.classList.add('arm'); btn.textContent = label;
    setTimeout(() => { if (btn.classList.contains('arm')) { btn.classList.remove('arm'); btn.textContent = btn.dataset.label; } }, 4000);
    return;
  }
  btn.classList.remove('arm'); btn.textContent = btn.dataset.label; fn();
}
$('memEpisodes').onclick = async (e) => {
  const id = e.target.dataset && e.target.dataset.ep; if (!id || !MEM.p) return;
  try { await memPost(`/api/memory/${encodeURIComponent(MEM.p.id)}/episodes`, { id, action: 'delete' }); loadMemCounts(); }
  catch (err) { toast(`not deleted: ${err.message || err}`, '#f87171'); }
};
$('memReview').onclick = async (e) => {
  const v = e.target.dataset && e.target.dataset.rv; if (!v || !MEM.p) return;
  const [kind, id, action] = v.split('|');
  try { await memPost(`/api/memory/${encodeURIComponent(MEM.p.id)}/review`, { kind, id, action }); loadMemCounts(); }
  catch (err) { toast(`not changed: ${err.message || err}`, '#f87171'); }
};
$('memFacts').onclick = async (e) => {
  const d = e.target.dataset || {}; const id = d.fc || d.fd; if (!id) return;
  try { await memPost('/api/memory_state/facts', { id, action: d.fc ? 'confirm' : 'delete' }); loadMemShared(); }
  catch (err) { toast(`not changed: ${err.message || err}`, '#f87171'); }
};
for (const [box, key] of [['memUse', 'use_memories'], ['memPause', 'paused'], ['memCloud', 'cloud_allowed']]) {
  $(box).onchange = async () => {
    try { await memPost('/api/memory_state/settings', { key, value: $(box).checked }); }
    catch (err) { toast(err.message || String(err), '#f87171'); }
    loadMemShared();
  };
}
$('memProcess').onclick = async () => {
  try {
    if ($('memProcess').dataset.tagger) { await memPost('/api/memory_state/tagger'); toast('downloading the name tagger…'); }
    else { await memPost('/api/memory_state/process'); toast('making memories from finished calls…'); }
  } catch (err) { toast(err.message || String(err), '#f87171'); }
  setTimeout(loadMemShared, 800);
};
$('memAllowSave').onclick = async () => {
  try { await memPost('/api/memory_state/allow', { names: $('memAllow').value.split('\n') }); toast('names saved'); }
  catch (err) { toast(`not saved: ${err.message || err}`, '#f87171'); }
};
$('memClearFacts').onclick = () => armTwice($('memClearFacts'), 'Click again to clear', async () => {
  try { await memPost('/api/memory_state/facts', { action: 'clear' }); loadMemShared(); } catch (err) { toast(String(err.message || err), '#f87171'); }
});
$('memOld').onclick = () => armTwice($('memOld'), 'Click again to delete', async () => {
  try { const m = await memPost('/api/memory_state/old_data'); toast(`deleted ${m.deleted} old file${m.deleted === 1 ? '' : 's'}`); loadMemShared(); }
  catch (err) { toast(String(err.message || err), '#f87171'); }
});
document.querySelector('#memModal .memmore').addEventListener('toggle', (e) => { if (e.target.open) loadMemLog(); });
function openMemory(p) {
  MEM.p = p;
  $('memTitle').textContent = `${p.name || p.id} · memory`;
  $('memTitle').style.color = p.accent || '';
  $('memDelAgent').style.display = p.custom ? '' : 'none';
  $('memWindow').value = '1h'; memHint();
  $('memCounts').innerHTML = '<span>loading…</span>';
  $('memModal').classList.add('open');
  loadMemCounts();
}
function closeMemory() { disarmWipe(); $('memModal').classList.remove('open'); MEM.p = null; }
async function wipeMemory() {
  const b = $('memWipe'); const p = MEM.p; if (!p) return;
  if (!b.classList.contains('arm')) {            // two-step: first click arms, second deletes
    b.classList.add('arm'); b.textContent = 'Click again to delete';
    MEM.armT = setTimeout(disarmWipe, 4000); return;
  }
  disarmWipe(); b.disabled = true;
  const w = $('memWindow').value;
  try {
    const r = await fetch(`/api/memory/${encodeURIComponent(p.id)}/wipe`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ window: w }) });
    const m = await r.json();
    if (!r.ok) throw new Error(m.error || r.status);
    const n = (m.episodes || 0) + (m.lessons || 0);
    toast(n ? `${p.name}: deleted ${n} memor${n === 1 ? 'y' : 'ies'} from ${MEM_LABEL[w]}` : `${p.name}: nothing to delete from ${MEM_LABEL[w]}`, p.accent);
    loadMemCounts();
  } catch (e) { toast(`memory delete failed: ${e.message || e}`, '#f87171'); }
  b.disabled = false;
}
$('memWindow').onchange = memHint;
$('memWipe').onclick = wipeMemory;
$('memClose').onclick = closeMemory; $('memDone').onclick = closeMemory;
$('memOpen').onclick = async () => { if (!MEM.p) return; const r = await fetch(`/api/memory/${encodeURIComponent(MEM.p.id)}/open`, { method: 'POST' }); if (!r.ok) toast('could not open the folder', '#f87171'); };
$('memDelAgent').onclick = async () => { const p = MEM.p; if (!p) return; closeMemory(); await deleteAgent(p); };
$('memModal').onclick = (e) => { if (e.target.id === 'memModal') closeMemory(); };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && $('memModal').classList.contains('open')) closeMemory(); });

/* ---------------------------------------------------------------- agent creator */
const CR = { step: 1, draft: null, accent: null, palette: [], usedVoices: [], previewT: 0 };
const crFields = ['identity', 'who', 'talk', 'bait', 'canon'];
function crPaintAv() {
  const name = $('crName').value.trim() || '?';
  const ac = CR.accent || (CR.draft && CR.draft.accent) || '#22d3ee';
  for (const id of ['crAv1', 'crAv2']) { const a = $(id); a.textContent = name.charAt(0).toUpperCase(); a.style.setProperty('--pc', ac); }
}
async function openCreator() {
  hideTip();
  CR.step = 1; CR.draft = null; CR.accent = null;
  $('crErr').textContent = ''; $('crName').value = ''; $('crDesc').value = '';
  try {
    const t = await (await fetch('/api/agents/template')).json();
    CR.palette = t.palette || []; CR.usedVoices = t.used_voices || [];
    const vs = $('crVoice'); vs.innerHTML = '';
    (t.voices || S.voices).forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v + (CR.usedVoices.includes(v) ? '  (in use)' : ''); vs.appendChild(o); });
    const free = (t.voices || []).find((v) => !CR.usedVoices.includes(v)); if (free) vs.value = free;
  } catch (e) { $('crErr').textContent = 'backend unreachable'; }
  crVoiceHint(); crShow(1); crPaintAv();
  $('creatorModal').classList.add('open'); setTimeout(() => $('crName').focus(), 50);
}
function crVoiceHint() { const v = $('crVoice').value; $('crVoiceHint').textContent = CR.usedVoices.includes(v) ? 'another agent already uses this voice' : ''; }
function closeCreator() { $('creatorModal').classList.remove('open'); }
function crShow(n) {
  CR.step = n;
  $('crStep1').style.display = n === 1 ? '' : 'none'; $('crStep2').style.display = n === 2 ? '' : 'none';
  $('stp1').classList.toggle('on', n === 1); $('stp2').classList.toggle('on', n === 2);
  $('crNext').style.display = n === 1 ? '' : 'none'; $('crSkip').style.display = n === 1 ? '' : 'none';
  $('crBack').style.display = n === 2 ? '' : 'none'; $('crCreate').style.display = n === 2 ? '' : 'none';
}
function crCheck1() {
  const name = $('crName').value.trim(), desc = $('crDesc').value.trim();
  if (name.length < 2) return 'Give the agent a name.';
  if (S.personas.some((p) => p.id === name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''))) return `An agent called ${name} already exists.`;
  if (desc.length < 10) return 'Describe them in at least a sentence.';
  return '';
}
async function crInterpret(useModel) {
  const err = crCheck1(); $('crErr').textContent = err; if (err) return;
  const name = $('crName').value.trim(), desc = $('crDesc').value.trim(), voice = $('crVoice').value;
  if (!useModel) {
    CR.draft = { name, voice, role: 'custom · agent', accent: null, source: 'manual', fields: { identity: '', who: desc, talk: '', bait: '', canon: '' } };
    return crFill();
  }
  $('crBusy').style.display = ''; $('crNext').disabled = true; $('crSkip').disabled = true;
  try {
    const r = await fetch('/api/agents/draft', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, description: desc, voice }) });
    const m = await r.json();
    if (!r.ok) throw new Error(m.error || 'draft failed');
    CR.draft = m; crFill();
  } catch (e) { $('crErr').textContent = String(e.message || e); }
  finally { $('crBusy').style.display = 'none'; $('crNext').disabled = false; $('crSkip').disabled = false; }
}
function crFill() {
  const d = CR.draft; CR.accent = d.accent || CR.accent || CR.palette[0] || '#22d3ee';
  $('crName2').textContent = d.name || $('crName').value;
  $('crSrc').textContent = d.source === 'llm' ? 'Drafted by the model' : d.source === 'fallback' ? 'Model failed: edit by hand' : d.source === 'heart' ? 'From heart.md' : 'Written by hand';
  if (d.name) $('crName').value = d.name;
  if (d.voice && [...$('crVoice').options].some((o) => o.value === d.voice)) $('crVoice').value = d.voice;
  $('crInterests').value = Array.isArray(d.interests) ? d.interests.join(', ') : (d.interests || (d.fields && d.fields.interests) || '');
  $('crTalk').value = d.talkativeness || 0.35; crTalkLabel();
  CR.promptEdited = !!d.prompt_locked;
  if (d.prompt_locked) { $('crPrompt').value = d.prompt; crPromptHint(); }
  $('crRole').value = d.role || '';
  crFields.forEach((k) => { $('f_' + k).value = (d.fields && d.fields[k]) || ''; });
  const sw = $('crSw'); sw.innerHTML = '';
  CR.palette.forEach((c) => { const s = document.createElement('div'); s.className = 'sw' + (c === CR.accent ? ' on' : ''); s.style.background = c; s.onclick = () => { CR.accent = c; crFill2(); }; sw.appendChild(s); });
  crShow(2); crPaintAv(); crPreview();
}
function crFill2() { for (const s of $('crSw').children) s.classList.toggle('on', s.style.background && hexOf(s.style.background) === CR.accent.toLowerCase()); crPaintAv(); }
function hexOf(rgb) { const m = rgb.match(/\d+/g); return m ? '#' + m.slice(0, 3).map((x) => (+x).toString(16).padStart(2, '0')).join('') : rgb; }
function crCurrentFields() { const f = {}; crFields.forEach((k) => { f[k] = $('f_' + k).value; }); return f; }
function crTalkLabel() { const v = +$('crTalk').value; $('crTalkV').textContent = v < 0.25 ? '· quiet' : v < 0.45 ? '· normal' : v < 0.65 ? '· chatty' : '· loud'; }
function crPromptHint() { $('crPromptHint').textContent = CR.promptEdited ? 'edited by hand — field changes no longer update it' : ''; }
function crPreview() {
  if (CR.promptEdited) return;
  clearTimeout(CR.previewT);
  CR.previewT = setTimeout(async () => {
    try {
      const r = await fetch('/api/agents/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: $('crName').value.trim(), fields: crCurrentFields() }) });
      const m = await r.json(); if (!CR.promptEdited) $('crPrompt').value = m.prompt || '';
    } catch (e) { /* keep last preview */ }
  }, 180);
}
async function crCreate() {
  $('crErr').textContent = '';
  if (!CR.promptEdited && $('f_who').value.trim().length < 10) { $('crErr').textContent = '"Who they are" is empty.'; return; }
  $('crCreate').disabled = true;
  try {
    const body = { name: $('crName').value.trim(), voice: $('crVoice').value, role: $('crRole').value, accent: CR.accent, fields: crCurrentFields(),
      interests: $('crInterests').value, talkativeness: +$('crTalk').value, prompt: CR.promptEdited ? $('crPrompt').value : '' };
    const r = await fetch('/api/agents', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const m = await r.json();
    if (!r.ok) throw new Error(m.error || 'create failed');
    const cur = S.persona; applyPersonasMsg(m); S.persona = cur;
    closeCreator(); toast(`created ${m.agent.name}`, m.agent.accent);
    setPersona(m.agent.id);
  } catch (e) { $('crErr').textContent = String(e.message || e); }
  finally { $('crCreate').disabled = false; }
}
$('crClose').onclick = closeCreator;
$('creatorModal').onclick = (e) => { if (e.target.id === 'creatorModal') closeCreator(); };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && $('creatorModal').classList.contains('open')) closeCreator(); });
$('crNext').onclick = () => crInterpret(true);
$('crSkip').onclick = () => crInterpret(false);
$('crBack').onclick = () => crShow(1);
$('crCreate').onclick = crCreate;
$('crName').oninput = () => { $('crErr').textContent = ''; crPaintAv(); };
$('crVoice').onchange = crVoiceHint;
crFields.forEach((k) => { $('f_' + k).oninput = crPreview; });
$('crDesc').onkeydown = (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) crInterpret(true); };
$('crTalk').oninput = crTalkLabel;
$('crPrompt').oninput = () => { CR.promptEdited = true; crPromptHint(); };
$('crRegen').onclick = (e) => { e.preventDefault(); CR.promptEdited = false; crPromptHint(); crPreview(); };
$('crHeart').onclick = (e) => { e.preventDefault(); $('crHeartFile').value = ''; $('crHeartFile').click(); };
$('crHeartFile').onchange = async () => {
  const f = $('crHeartFile').files[0]; if (!f) return;
  $('crErr').textContent = '';
  try {
    const text = await f.text();
    const r = await fetch('/api/agents/heart', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, name: $('crName').value.trim(), voice: $('crVoice').value }) });
    const m = await r.json(); if (!r.ok) throw new Error(m.error || 'import failed');
    if (!m.name) m.name = $('crName').value.trim() || f.name.replace(/\.(heart\.)?(md|txt)$/i, '');
    CR.draft = m; crFill();
  } catch (e) { $('crErr').textContent = String(e.message || e); }
};
$('crVoiceUp').onclick = () => { $('crVoiceFile').value = ''; $('crVoiceFile').click(); };
$('crVoiceFile').onchange = async () => {
  const f = $('crVoiceFile').files[0]; if (!f) return;
  const label = prompt('Name this voice', f.name.replace(/\.[^.]+$/, '').slice(0, 24)); if (!label) return;
  const transcript = prompt('Optional: type exactly what is said in the clip (improves cloning). Leave blank to skip.', '') || '';
  $('crErr').textContent = ''; $('crVoiceHint').textContent = 'importing…';
  try {
    const q = new URLSearchParams({ label, filename: f.name, transcript });
    const r = await fetch('/api/voices/import?' + q, { method: 'POST', body: f });
    const m = await r.json(); if (!r.ok) throw new Error(m.error || 'import failed');
    const vs = $('crVoice'); const o = document.createElement('option'); o.value = m.key; o.textContent = m.key; vs.appendChild(o); vs.value = m.key;
    S.voices = m.voices || S.voices;
    $('crVoiceHint').textContent = `imported ${m.seconds}s clip`;
  } catch (e) { $('crVoiceHint').textContent = ''; $('crErr').textContent = String(e.message || e); }
};

/* ---------------------------------------------------------------- chat */
let chatKey = 0; const chatEls = new Map(); let chatEmptyEl = null; const typingEls = { user: null, assistant: null };
function buildMsg(m, streaming) {
  const el = document.createElement('div');
  if (m.role === 'sys') { el.className = 'msg sys'; el.textContent = m.content; return el; }
  el.className = `msg ${m.role}${streaming ? ' streaming' : ''}`;
  if (m.role === 'assistant') {
    const p = persona(m.persona || S.persona); el.style.setProperty('--pc', p.accent);
    el.innerHTML = `<div class="who"><span class="t">${m.time || ''}</span>${esc(p.name)}</div><div class="body"></div>`;
  } else {
    el.style.setProperty('--sc', spkColor(m.label));
    el.innerHTML = `<div class="who">${esc(m.label || 'You')}<span class="t">${m.time || ''}</span></div><div class="body"></div>`;
  }
  el.querySelector('.body').textContent = m.content;
  return el;
}
function setTyping(kind, msg) {
  const feed = $('feed'); let el = typingEls[kind];
  if (!msg) { if (el) { el.remove(); typingEls[kind] = null; } return; }
  const sig = kind + '|' + (msg.label || msg.persona || '');
  if (!el || el.dataset.sig !== sig) { if (el) el.remove(); el = buildMsg(msg, true); el.dataset.sig = sig; typingEls[kind] = el; }
  else el.querySelector('.body').textContent = msg.content;
  feed.appendChild(el);   // re-append keeps typing bubbles last; no re-animation (node is moved, not recreated)
}
// Incremental: finished messages are built ONCE (keyed), so their fade-in runs once.
// Rebuilding the feed on every partial restarted every animation -> feed stuck at opacity 0.
function renderChat() {
  const feed = $('feed');
  const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 80;
  const shown = S.chat.slice(-160);
  shown.forEach((m) => { if (m._k == null) m._k = ++chatKey; });
  const keep = new Set(shown.map((m) => m._k));
  for (const [k, el] of chatEls) if (!keep.has(k)) { el.remove(); chatEls.delete(k); }
  const empty = !S.chat.length && !S.typingUser && !S.typingAssistant;
  if (empty && !chatEmptyEl) { chatEmptyEl = document.createElement('div'); chatEmptyEl.className = 'empty-chat'; chatEmptyEl.innerHTML = 'No messages yet<br>talk in the call — transcripts stream here live'; feed.appendChild(chatEmptyEl); }
  if (!empty && chatEmptyEl) { chatEmptyEl.remove(); chatEmptyEl = null; }
  let prev = null;   // keep DOM order == chat order (backlog can prepend)
  for (const m of shown) {
    let el = chatEls.get(m._k);
    if (!el) { el = buildMsg(m, false); chatEls.set(m._k, el); }
    const want = prev ? prev.nextSibling : feed.firstChild;
    if (el !== want) feed.insertBefore(el, want);
    prev = el;
  }
  setTyping('user', S.typingUser ? { role: 'user', content: S.typingUser, label: S.typingUserSpk || '…' } : null);
  setTyping('assistant', S.typingAssistant ? { role: 'assistant', content: S.typingAssistant, persona: S.persona } : null);
  if (atBottom) feed.scrollTop = feed.scrollHeight;
  $('chatCount').textContent = `${S.chat.filter((m) => m.role !== 'sys').length} msgs`;
}
// Live transcripts arrive several times a second: draw the chat at most once per frame, and let a
// partial touch only its typing line (owner 11-02: "so many things updating at once it kinda lags").
let chatQueued = false, chatFull = false;
function queueChat(full) {
  chatFull = chatFull || full;
  if (chatQueued) return;
  chatQueued = true;
  requestAnimationFrame(() => { chatQueued = false; const f = chatFull || !!chatEmptyEl; chatFull = false; f ? renderChat() : renderTyping(); });
}
function renderTyping() {
  const feed = $('feed');
  const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 80;
  setTyping('user', S.typingUser ? { role: 'user', content: S.typingUser, label: S.typingUserSpk || '…' } : null);
  setTyping('assistant', S.typingAssistant ? { role: 'assistant', content: S.typingAssistant, persona: S.persona } : null);
  if (atBottom) feed.scrollTop = feed.scrollHeight;
}
const CHAT_KEEP = 400;   // the feed shows 160; older history only cost time on every redraw
function pushChat(m) { S.chat.push(m); if (S.chat.length > CHAT_KEEP) S.chat.splice(0, S.chat.length - CHAT_KEEP); }
$('clearView').onclick = () => { S.chat = []; renderChat(); };
$('clearBtn').onclick = async () => {
  const b = $('clearBtn'); b.disabled = true;
  const { status, data } = await postAction('/api/clear_history');
  b.disabled = false;
  if (status === 200) { S.chat.push({ role: 'sys', content: '— model history cleared —' }); renderChat(); toast('history cleared'); }
  else toast(status === 0 ? "the backend didn't answer" : (data.error || `clear history failed (${status})`), '#f87171');
};
const speedWord = (v) => (v <= 10 ? 'Snappy' : v <= 40 ? 'Quick' : v <= 70 ? 'Relaxed' : 'Patient');
function setSpeedUi(v) { $('speed').value = v; $('speedV').textContent = speedWord(+v); $('speed').title = `${v}/100 — lower = jumps in faster`; }
$('speed').oninput = (e) => setSpeedUi(+e.target.value);
$('speed').onchange = (e) => { send({ type: 'set_speed', speed: +e.target.value }); toast(`turn-taking → ${speedWord(+e.target.value).toLowerCase()}`); };
setSpeedUi(0);

/* ---------------------------------------------------------------- /ws control */
let ws = null, wsRetry = 0;
function send(m) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(m)); }
function connectWs() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { S.wsOpen = true; wsRetry = 0; send({ type: 'list_personas' }); send({ type: 'list_devices' }); maybeStartBrowserAudio(); };
  ws.onmessage = (e) => { if (typeof e.data === 'string') { try { onWs(JSON.parse(e.data)); } catch (err) { console.error(err); } } };
  ws.onclose = () => { S.wsOpen = false; stopBrowserAudio(); setTimeout(connectWs, Math.min(8000, 800 * ++wsRetry)); };
  ws.onerror = () => {};
}
function onWs(m) {
  const c = m.content ?? '';
  switch (m.type) {
    case 'partial_user_request': S.typingUser = c; S.typingUserSpk = m.speakerId || S.typingUserSpk; queueChat(false); break;
    case 'final_user_request':
      if (c.trim()) pushChat({ role: 'user', content: c, label: spkLabel(m), time: hhmm() });
      if (guideEl && c.trim()) guideEl.classList.add('hidden'); // a live conversation needs no 'next step' card
      S.typingUser = ''; S.typingUserSpk = ''; queueChat(true); break;
    case 'partial_assistant_answer': S.typingAssistant = c; queueChat(false); break;
    case 'final_assistant_answer':
      if (c.trim()) pushChat({ role: 'assistant', content: c, persona: m.personaId || S.persona, time: hhmm() });
      S.typingAssistant = ''; queueChat(true); break;
    case 'personas':
      applyPersonasMsg(m); break;
    case 'chat_backlog': {
      const seen = new Set(S.chat.map((x) => x.role + '|' + x.content));
      const back = (m.items || []).map((x) => x.type === 'final_user_request'
        ? { role: 'user', content: x.content || '', label: spkLabel(x), time: hhmm(x.t) }
        : { role: 'assistant', content: x.content || '', persona: x.personaId || S.persona, time: hhmm(x.t) })
        .filter((x) => x.content.trim() && !seen.has(x.role + '|' + x.content));
      if (back.length) S.chat = back.concat(S.chat);
      if (typeof m.speed === 'number') setSpeedUi(m.speed);
      renderChat(); break;
    }
    case 'devices': renderDevices(m); break;
    case 'action_error': toast(`${(m.action || 'action').replace('_', ' ')} failed: ${m.error}`, '#f87171'); break;
    case 'tts_chunk': playBrowserTts(c); break;
    case 'stop_tts': case 'tts_interruption': if (ttsNode) ttsNode.port.postMessage({ type: 'clear' }); break;
  }
}
function renderDevices(info) {
  if (!info.devices) return;
  const fill = (sel, list, cur) => { sel.innerHTML = ''; list.forEach((d) => { const o = document.createElement('option'); o.value = d.index; o.textContent = `${d.index} · ${d.name}`; o.selected = d.index === cur; sel.appendChild(o); }); };
  fill($('devIn'), info.devices.inputs, info.current.input); fill($('devOut'), info.devices.outputs, info.current.output);
  const inN = info.devices.inputs.find((d) => d.index === info.current.input), outN = info.devices.outputs.find((d) => d.index === info.current.output);
  if (info.virtual) { $('devLabel').textContent = info.virtual; return; }
  $('devLabel').textContent = `${(inN?.name || '?').replace(/\(.*$/, '').trim()} → ${(outN?.name || '?').replace(/\(.*$/, '').trim()}`;
}
const pushDevices = async () => {
  const { status, data } = await postAction('/api/audio_devices', { input: +$('devIn').value, output: +$('devOut').value });
  if (status === 200) { renderDevices(data); toast('audio routing updated'); }
  else { toast(status === 0 ? "the backend didn't answer" : `audio devices: ${data.error || status}`, '#f87171'); send({ type: 'list_devices' }); }
};
$('devIn').onchange = pushDevices; $('devOut').onchange = pushDevices;

/* browser-mode audio (only when no call bridge is running) */
let actx = null, micNode = null, ttsNode = null, micStream = null, batch = null, bView = null, bI16 = null, bOff = 0;
async function maybeStartBrowserAudio() {
  if (S.bridge || !S.snap || micNode) return;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    actx = actx || new AudioContext();
    await actx.audioWorklet.addModule('/static/pcmWorkletProcessor.js');
    await actx.audioWorklet.addModule('/static/ttsPlaybackProcessor.js');
    micNode = new AudioWorkletNode(actx, 'pcm-worklet-processor');
    ttsNode = new AudioWorkletNode(actx, 'tts-playback-processor'); ttsNode.connect(actx.destination);
    micNode.port.onmessage = ({ data }) => {
      const inc = new Int16Array(data); let r = 0;
      while (r < inc.length) {
        if (!batch) { batch = new ArrayBuffer(8 + 4096); bView = new DataView(batch); bI16 = new Int16Array(batch, 8); bOff = 0; }
        const n = Math.min(inc.length - r, 2048 - bOff); bI16.set(inc.subarray(r, r + n), bOff); bOff += n; r += n;
        if (bOff === 2048) { bView.setUint32(0, Date.now() & 0xffffffff); bView.setUint32(4, 0); if (ws?.readyState === 1) ws.send(batch); batch = null; }
      }
    };
    actx.createMediaStreamSource(micStream).connect(micNode);
  } catch (e) { console.warn('browser audio unavailable', e); }
}
function stopBrowserAudio() { micNode?.disconnect(); ttsNode?.disconnect(); micStream?.getTracks().forEach((t) => t.stop()); micNode = ttsNode = micStream = null; }
function playBrowserTts(b64) { if (!ttsNode) return; const raw = atob(b64); const u = new Uint8Array(raw.length); for (let i = 0; i < raw.length; i++) u[i] = raw.charCodeAt(i); ttsNode.port.postMessage(new Int16Array(u.buffer)); }

/* ---------------------------------------------------------------- /telemetry */
let tel = null, telRetry = 0;
function connectTel() {
  tel = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/telemetry`);
  tel.onopen = () => { S.telOpen = true; telRetry = 0; $('overlay').classList.remove('show'); };
  tel.onmessage = (e) => { try { onTel(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  tel.onclose = () => { S.telOpen = false; $('overlay').classList.add('show'); $('ovMsg').textContent = `reconnecting… (attempt ${telRetry + 1})`; setTimeout(connectTel, Math.min(6000, 700 * ++telRetry)); };
  tel.onerror = () => {};
}
function onTel(f) {
  if (f.type === 'fast') {
    S.fast = f;
    S.scope.push([f.in || 0, f.out || 0]); if (S.scope.length > SCOPE_N) S.scope.shift();
    dirty.scope = true;
    updatePartial(); updateLamps(); return;
  }
  if (f.type === 'snap') {
    S.snap = f; S.bridge = !!f.bridge; renderSnap(f); dirty.snap = true;
    // Snaps from before the call bridge exists (backend still starting) opened the browser mic, and it
    // stayed open for the whole session: the server ignores that audio in bridge mode (11-02).
    if (S.bridge && micNode) stopBrowserAudio();
    else if (S.wsOpen) maybeStartBrowserAudio();
    return;
  }
  if (f.type === 'events') { f.items.forEach(onEvent); return; }
  if (f.type === 'lines') { appendLogs(f.items); }
}

/* ---------------------------------------------------------------- core column */
const db = (x) => (x > 1e-6 ? 20 * Math.log10(x) : -99);
// DOM writes only when the value changed: an unchanged write still costs a style pass.
function setText(el, txt) { txt = String(txt); if (el && el.textContent !== txt) el.textContent = txt; }
function setHTML(el, html) { if (el && el._html !== html) { el._html = html; el.innerHTML = html; } }
// A folded side panel isn't drawn or rebuilt at all.
const shown = (el) => !!el && !el.closest('.panel.collapsed');
const meterEls = {};
function meter(id, vid, x, now) {   // every frame: the bar is a compositor transform, the dB text 10 times a second
  const m = meterEls[id] || (meterEls[id] = { bar: $(id), txt: $(vid), v: -1, at: 0, idle: null });
  const d = db(x), v = clamp((d + 60) / 60);
  if (Math.abs(v - m.v) > .002) { m.v = v; m.bar.style.transform = `scaleX(${v.toFixed(3)})`; }
  if (now - m.at < 100) return;
  m.at = now;
  const idle = d < -60;
  setText(m.txt, idle ? 'idle' : `${d.toFixed(0)} dB`);
  if (idle !== m.idle) { m.idle = idle; m.txt.classList.toggle('idle', idle); }
}
function updatePartial() {
  const t = S.fast.partial || S.typingUser;
  const el = $('partialTxt');
  if (t && (S.fast.recording || S.typingUser)) { setText(el, t); el.classList.remove('empty'); }
  else if (!S.fast.recording) { setText(el, 'waiting for speech…'); el.classList.add('empty'); }
}
let pipeEls = null, lastPipeKey = '', lastLampKey = '';
function updateLamps() {
  const f = S.fast, fl = S.snap?.pipeline?.floor;
  const st = f.speaking ? 'Speaking' : f.thinking ? 'Thinking' : f.recording ? 'Listening' : fl?.quiet ? 'Quiet mode' : S.telOpen ? 'Ready' : 'Offline';
  const lk = `${!!f.recording}${!!f.thinking}${!!f.speaking}${!!fl?.quiet}${st}`;
  if (lk !== lastLampKey) {
    lastLampKey = lk;
    $('lHear').classList.toggle('on', !!f.recording);
    $('lThink').classList.toggle('on', !!f.thinking);
    $('lSpeak').classList.toggle('on', !!f.speaking);
    $('lQuiet').classList.toggle('on', !!fl?.quiet);
    $('orbState').textContent = st;
  }
  // pipeline stepper
  const now = performance.now();
  const gen = S.snap?.pipeline?.gen;
  let stage = null, done = [];
  if (f.speaking) { stage = 'speak'; done = ['hear', 'end', 'decide', 'gen']; }
  else if (f.thinking && gen?.decision === 'SPEAK') { stage = 'gen'; done = ['hear', 'end', 'decide']; }
  else if (f.thinking) { stage = 'decide'; done = f.recording ? ['hear'] : ['hear', 'end']; }
  else if (now - S.turnEndAt < 900) { stage = 'end'; done = ['hear']; }
  else if (f.recording) stage = 'hear';
  const key = stage + '|' + done.join(',');
  if (key === lastPipeKey) return;     // ~15 telemetry frames/s: touch the stepper only when it changes
  lastPipeKey = key;
  (pipeEls ||= [...document.querySelectorAll('#pipe div')]).forEach((d) => { d.classList.toggle('on', d.dataset.s === stage); d.classList.toggle('done', done.includes(d.dataset.s)); });
}

/* ---------------------------------------------------------------- snapshot panels */
function fmt(v, unit, dig = 0) { return v == null || Number.isNaN(v) ? '—' : `${(+v).toFixed(dig)}<small>${unit}</small>`; }
const series = {};
function renderSnap(s) {
  // header pills + model
  const up = (id, cls) => { $(id).className = `pill ${cls}`; };
  up('pBackend', 'ok'); up('pOllama', s.ollama?.up ? 'ok' : 'bad');
  up('pBridge', s.bridge ? 'ok' : 'warn'); up('pWs', S.wsOpen ? 'ok' : 'bad');
  const po = s.bridge && s.bridge.playout, vs = $('vStutter');
  if (vs) {
    if (!po) { setText(vs, '—'); vs.style.color = ''; }
    else {
      const last = po.last_reply_underruns || 0;
      setText(vs, `${po.underruns} gaps / ${po.replies} replies · ${Math.round(po.silence_ms)}ms · buf ${po.prebuffer_ms}ms`);
      vs.style.color = last ? 'rgb(251,191,36)' : (po.underruns ? '' : 'rgb(52,211,153)');
    }
  }
  setText($('pBridge').lastChild, s.bridge ? 'Call' : 'Browser');
  renderEyes(s.screen);
  renderOwner(s.owner);
  renderBrain(s.llm_backend);
  const u = Math.floor(s.uptime || 0); setText($('uptime'), `${String(Math.floor(u / 3600)).padStart(2, '0')}:${String(Math.floor(u / 60) % 60).padStart(2, '0')}:${String(u % 60).padStart(2, '0')}`);
  const pl = s.pipeline || {};
  setText($('modelName'), (s.llm_backend?.mode === 'remote' && s.llm_backend.remote) ? `${s.llm_backend.remote.model} @ remote` : (pl.model || '—'));
  if (pl.persona && pl.persona !== S.persona) applyCurrent({ persona: pl.persona, voice: pl.voice });   // the server's agent, always
  renderPersonaCards();

  // latency
  const L = s.latency || {};
  setHTML($('kE2E'), L.e2e_s?.n ? fmt(L.e2e_s.p50, 's', 2) : '—');
  setHTML($('kHdr'), L.header_ms?.n ? fmt(L.header_ms.p50, 'ms') : '—');
  setHTML($('kPause'), L.pause_s?.n ? fmt(L.pause_s.p50, 's', 2) : '—');
  series.e2e = L.e2e_s?.series || []; series.hdr = L.header_ms?.series || []; series.pause = L.pause_s?.series || [];
  const C = s.counters || {};
  setText($('cSpeak'), C.decision_SPEAK || 0); setText($('cHold'), C.decision_HOLD || 0);
  setText($('cBad'), (C.decision_INVALID || 0) + (C.error || 0));
  setText($('turnsN'), `${L.e2e_s?.n || 0} replies · p90 ${L.e2e_s?.n ? L.e2e_s.p90.toFixed(2) + 's' : '—'}`);

  // gpu
  const g = s.gpu || {};
  if (!g.error) {
    ringT.util = (g.util || 0) / 100; ringT.mem = (g.mem_used || 0) / (g.mem_total || 1);
    ringLbl.util = `${g.util ?? 0}%`; ringLbl.mem = `${(g.mem_used || 0).toFixed(1)}G`;
    setText($('gpuName'), (g.name || '').replace('NVIDIA GeForce ', ''));
    if (shown($('gpuStats'))) setHTML($('gpuStats'), `temp <b>${g.temp}°C</b><br>power <b>${(g.power || 0).toFixed(0)}</b>/${(g.power_limit || 0).toFixed(0)} W<br>clock <b>${g.clock}</b> MHz<br>free <b>${((g.mem_total || 0) - (g.mem_used || 0)).toFixed(1)}</b> GB`);
    if (shown($('gpuProcs'))) setHTML($('gpuProcs'), (g.procs || []).map((p) => `<div class="rl-item"><span class="n">${esc(p.name)}</span><span class="m">${p.gb.toFixed(2)} GB</span><div class="minibar"><b style="width:${clamp(p.gb / (g.mem_total || 24)) * 100}%"></b></div></div>`).join(''));
  } else { setText($('gpuStats'), g.error); }
  gpuHist = s.gpu_hist || [];

  // models + process
  const ms = s.ollama?.models || [];
  if (shown($('models'))) setHTML($('models'), ms.length ? ms.map((m) => {
    const exp = m.expires ? Math.max(0, (Date.parse(m.expires) - Date.now()) / 60000) : null;
    return `<div class="rl-item"><span class="n">${esc(m.name)}</span><span class="m">${m.vram_gb.toFixed(1)} GB · ${esc(m.params || '')} ${esc(m.quant || '')}</span><div class="minibar"><b style="width:${exp == null ? 0 : clamp(exp / 30) * 100}%"></b></div><span class="m" style="grid-column:1/-1">ctx ${m.ctx ?? '—'} · keep-alive ${exp == null ? '—' : exp.toFixed(0) + ' min'}</span></div>`;
  }).join('') : `<div class="stats">${s.ollama?.up ? 'no model resident' : 'ollama unreachable'}</div>`);
  const p = s.proc || {};
  setText($('procStats'), p.error ? '' : `cpu ${(p.sys_cpu || 0).toFixed(0)}% · ram ${(p.rss_gb || 0).toFixed(1)}G · ${p.threads} thr`);

  // room
  const fl = pl.floor || {};
  setHTML($('floorRow'), [
    `<span class="chip ${fl.partner ? 'on' : ''}">partner ${esc((pl.names || {})[fl.partner] || cachedName(fl.partner) || fl.partner || 'none')}</span>`,
    `<span class="chip ${fl.quiet ? 'q' : ''}">${fl.quiet ? `quiet ${Math.ceil(fl.quiet_left)}s` : 'open floor'}</span>`,
    `<span class="chip">room ${fl.room_size ?? 0}</span>`,
    `<span class="chip">history ${pl.history_len ?? 0}</span>`,
  ].join(''));
  const roster = pl.roster || [];
  const dz = Object.fromEntries((s.speakers || []).map((x) => [x.label, x]));
  const liveSpk = S.typingUserSpk;
  setText($('roomN'), `${roster.length} people · ${(s.speakers || []).length} voices`);
  // The "ago" times change every second; the rows are rebuilt only when something else changes, and the times are
  // then updated in place (a full rebuild every second cost about 4% of the main thread on a wide window).
  const agos = roster.map((r) => (r.idle_s < 60 ? r.idle_s.toFixed(0) + 's' : Math.round(r.idle_s / 60) + 'm'));
  if (shown($('roster'))) setHTML($('roster'), roster.length ? roster.map((r, i) => {
    const col = spkColor(r.id), d = dz[r.id];
    const nm = r.name || cachedName(r.id);
    const ov = s.owner && s.owner.voice, mine = !!(ov && (ov.labels || []).includes(r.id));
    const tags = [mine ? '<span class="tag you">you</span>' : '',
                  r.addressed ? `<span class="tag hot">called ${r.addressed}×</span>` : ''].join('');
    const meBtns = !ov ? '' : mine
      ? `<span class="mebtn" data-notme="${esc(r.id)}" title="This voice isn't you (until Atlas restarts)">Not me</span><span class="mebtn" data-forgetme="1" title="Delete your saved voice">Forget my voice</span>`
      : `<span class="mebtn" data-me="${esc(r.id)}" title="Every agent will recognize this voice as you in new calls">This is me</span>`;
    const muted = (s.owner?.muted_speakers || []).includes(String(r.id).toUpperCase());
    const meta = `${r.turns} turns${d ? ` · ${d.speech_s}s` : ''} · <span class="ago" data-i="${i}"></span>`;
    return `<div class="spk ${r.id === liveSpk ? 'live' : ''} ${r.id === fl.partner ? 'partner' : ''} ${muted ? 'muted' : ''}" title="${esc(r.id)} · ${r.words} words">
      <div class="sav" style="--c:${col}">${esc((nm || r.id).slice(0, 2))}</div>
      <div class="sbody"><div class="sname"><span>${esc(nm || 'Unknown voice')}</span>${tags}<span class="smeta">${meta}</span></div><div class="line">${esc(r.last)}</div></div>
      <span class="sbtns">${meBtns}<span class="mutebtn ${muted ? 'on' : ''}" data-mute="${esc(r.id)}" title="${muted ? 'Unmute' : 'Ignore this speaker (agent never answers them)'}">${muted ? 'muted' : 'mute'}</span></span></div>`;
  }).join('') : `<div class="stats">${(s.speakers || []).map((x) => `${esc(x.label)} · ${x.speech_s}s`).join('<br>') || 'nobody has spoken yet'}</div>`);
  if (shown($('roster'))) for (const el of $('roster').querySelectorAll('.ago')) setText(el, agos[+el.dataset.i] || '');

  // threads
  const th = pl.threads || {};
  const open = th.open || [];
  if (shown($('threads'))) setHTML($('threads'), th.error ? `<div class="stats">${esc(th.error)}</div>` : (open.length ? open.map((t) => {
    const act = th.active && th.active.speaker === t.speaker && th.active.text === t.text;
    return `<div class="thread ${act ? 'active' : ''}"><div class="top"><span style="color:${spkColor(t.speaker)}">${esc(t.speaker)}${act ? ' · ANSWERING' : ''}</span><span>score ${t.score}</span></div><div>${esc(t.text)}</div><div class="rs">${(t.reasons || []).map((x) => `<span>${esc(x)}</span>`).join('')}</div></div>`;
  }).join('') : '<div class="stats">no open threads</div>'));

  // context
  const cv = pl.convo || {};
  setText($('ctxStats'), Object.keys(cv).length ? Object.entries(cv).slice(0, 4).map(([k, v]) => `${k} ${typeof v === 'number' ? Math.round(v) : v}`).join(' · ') : '—');
  setText($('roomNote'), pl.room_note || '—');
}

/* ---------------------------------------------------------------- events + logs */
const evBox = $('events'), logBox = $('logs');
function onEvent(e) {
  if (e.kind === 'turn_end') S.turnEndAt = performance.now();
  let k = e.kind, x = '', cls = e.kind;
  switch (e.kind) {
    case 'decision': k = e.action; cls = e.action; x = `${e.target ? '→ ' + e.target + ' · ' : ''}${Math.round(e.ms)} ms`; break;
    case 'user_final': k = e.speaker || 'user'; x = e.text; break;
    case 'agent_final': k = 'reply'; x = e.text; break;
    case 'latency': k = 'latency'; x = `${e.s.toFixed(2)} s speech→voice`; break;
    case 'ttft': x = `gen ${e.gen} · ${e.s.toFixed(2)} s`; break;
    case 'gen_start': k = 'gen'; x = e.text; break;
    case 'abort': x = e.text; break;
    case 'gate': x = e.text; break;
    case 'persona': x = e.name; break;
    case 'voice': x = e.name; break;
    case 'error': x = e.text; toast('pipeline error — see feed', 'var(--bad)'); break;
    case 'speaker': case 'turn_end': case 'tts_start': return;
    default: x = JSON.stringify(e).slice(0, 120);
  }
  evPending.push({ cls, tm: hhmm(e.t), k, x });
}
// Events and log lines arrive in bursts (~85 log lines a second in a heavy call): they are queued and put on the page
// once per frame, and log lines only while the log tab is open (they used to build hundreds of hidden rows a second).
const EV_KEEP = 120, LOG_KEEP = 300;
const evPending = [], logBuf = [];
let logShown = 0, logRebuild = true;
function appendLogs(items) {
  for (const l of items) logBuf.push(l);
  if (logBuf.length > LOG_KEEP) { const cut = logBuf.length - LOG_KEEP; logBuf.splice(0, cut); logShown = Math.max(0, logShown - cut); }
}
function logRow(l) { const d = document.createElement('div'); d.className = l.lvl; d.textContent = `${hhmm(l.t)} ${l.src.padEnd(12)} ${l.msg}`; return d; }
// at most 4 times a second: each flush moves every row of a visible list and repaints it (on a 3440 px window the
// event list alone cost 8% of the main thread when flushed every frame)
const FEED_FLUSH_MS = 250;
let lastFeedFlush = 0;
function flushFeeds() {
  const now = performance.now();
  if (now - lastFeedFlush < FEED_FLUSH_MS) return;
  lastFeedFlush = now;
  if (evPending.length && shown(evBox)) {
    const frag = document.createDocumentFragment();
    for (let i = evPending.length - 1; i >= Math.max(0, evPending.length - EV_KEEP); i--) {   // newest first
      const p = evPending[i], d = document.createElement('div');
      d.className = `ev ${p.cls}`; d.innerHTML = `<span class="tm">${p.tm}</span><span class="k">${esc(p.k)}</span><span class="x" title="${esc(p.x)}">${esc(p.x)}</span>`;
      frag.appendChild(d);
    }
    evPending.length = 0;
    evBox.prepend(frag);
    while (evBox.children.length > EV_KEEP) evBox.lastChild.remove();
  } else if (evPending.length > EV_KEEP) evPending.splice(0, evPending.length - EV_KEEP);
  if (S.evTab !== 'logs' || !shown(logBox) || (!logRebuild && logShown >= logBuf.length)) return;
  const frag = document.createDocumentFragment();
  if (logRebuild) { logBox.textContent = ''; logShown = 0; logRebuild = false; }
  for (let i = logShown; i < logBuf.length; i++) frag.appendChild(logRow(logBuf[i]));
  logShown = logBuf.length;
  logBox.appendChild(frag);
  while (logBox.children.length > LOG_KEEP) logBox.firstChild.remove();
  logBox.scrollTop = logBox.scrollHeight;
}
document.querySelectorAll('.tabs span').forEach((t) => t.onclick = () => {
  S.evTab = t.dataset.t; document.querySelectorAll('.tabs span').forEach((x) => x.classList.toggle('on', x === t));
  evBox.style.display = S.evTab === 'events' ? '' : 'none'; logBox.style.display = S.evTab === 'logs' ? '' : 'none';
  logRebuild = true;
});

/* ---------------------------------------------------------------- canvases */
// Canvas sizes come from a ResizeObserver, not getBoundingClientRect() on every frame: reading layout
// right after a chat update forced a full relayout ~30 times a second.
const canvasBox = new WeakMap();
const canvasRO = new ResizeObserver((es) => { for (const e of es) canvasBox.set(e.target, e.target.getBoundingClientRect()); });
function boxOf(c) { let r = canvasBox.get(c); if (!r) { r = c.getBoundingClientRect(); canvasBox.set(c, r); canvasRO.observe(c); } return r; }
function fit(c) { const r = boxOf(c), dpr = window.devicePixelRatio || 1; const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr)); if (c.width !== w || c.height !== h) { c.width = w; c.height = h; } return [c.getContext('2d'), w, h, dpr]; }

// orb: thin status ring. Spectrum ticks (call in), output-level arc, thinking sweep.
// Drawn every frame while anything moves, so it stays cheap: the 64 tick directions are computed once, and ticks
// are stroked in a few brightness steps (one path each) instead of 64 strokes with 64 colour strings.
const TICKS = 64, TICK_STEPS = 6;
const tickCos = Float32Array.from({ length: TICKS }, (_, i) => Math.cos((i / TICKS) * Math.PI * 2 - Math.PI / 2));
const tickSin = Float32Array.from({ length: TICKS }, (_, i) => Math.sin((i / TICKS) * Math.PI * 2 - Math.PI / 2));
function drawOrb(t) {
  const [g, w, h, dpr] = fit($('orb'));
  g.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h * .5, R = Math.min(w, h) * .36, sm = S.sm;
  const out = clamp((db(sm.out) + 55) / 50), inn = clamp((db(sm.clean) + 55) / 50);
  g.lineCap = 'round';
  g.lineWidth = 1 * dpr; g.strokeStyle = 'rgba(255,255,255,.07)';
  g.beginPath(); g.arc(cx, cy, R, 0, 7); g.stroke();
  // spectrum ticks around the ring
  const spec = sm.spec, half = spec.length, r0 = R + 5 * dpr, k = R * .32 * (0.4 + inn);
  g.lineWidth = 1.6 * dpr; g.strokeStyle = 'rgb(121,192,255)';
  for (let step = 0; step < TICK_STEPS; step++) {
    g.beginPath();
    let any = false;
    for (let i = 0; i < TICKS; i++) {
      const v = spec[i < half ? i : TICKS - 1 - i] || 0;
      if (Math.min(TICK_STEPS - 1, (v * TICK_STEPS) | 0) !== step) continue;
      const r1 = r0 + 2 * dpr + v * k;
      g.moveTo(cx + tickCos[i] * r0, cy + tickSin[i] * r0); g.lineTo(cx + tickCos[i] * r1, cy + tickSin[i] * r1);
      any = true;
    }
    if (any) { g.globalAlpha = .18 + ((step + .5) / TICK_STEPS) * .7; g.stroke(); }
  }
  g.globalAlpha = 1;
  // voice-out level as an arc on the ring
  if (out > .01) {
    g.lineWidth = 2.5 * dpr; g.strokeStyle = rgba(.95);
    g.beginPath(); g.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + out * Math.PI * 2); g.stroke();
  }
  // thinking: a short arc that sweeps
  if (sm.think > .02) {
    const a = t / 260;
    g.lineWidth = 2 * dpr; g.strokeStyle = `rgba(164,139,250,${sm.think})`;
    g.beginPath(); g.arc(cx, cy, R - 7 * dpr, a, a + 1.1); g.stroke();
  }
}
function drawSpec() {
  const [g, w, h] = fit($('spec')); g.clearRect(0, 0, w, h);
  const s = S.sm.spec, n = s.length, bw = w / n;
  g.fillStyle = 'rgb(121,192,255)';
  for (let i = 0; i < n; i++) {
    const v = s[i], bh = Math.max(2, v * h);
    g.globalAlpha = .25 + v * .6; g.fillRect(i * bw + 1, h - bh, bw - 2, bh);
  }
  g.globalAlpha = 1;
}
const SCOPE_N = 360;   // ~12 s of level history at 30 telemetry frames a second
function drawScope() {
  const [g, w, h] = fit($('scope')); g.clearRect(0, 0, w, h);
  const d = S.scope, n = d.length; if (n < 2) return;
  const line = (idx, col) => {
    g.beginPath();
    d.forEach((p, i) => { const x = (i / (SCOPE_N - 1)) * w, y = h - clamp((db(p[idx]) + 60) / 60) * (h - 2) - 1; i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
  };
  line(0, 'rgba(121,192,255,.8)'); line(1, rgba(.95));
}
function spark(id, data, col) {
  const [g, w, h] = fit($(id)); g.clearRect(0, 0, w, h);
  if (!data || data.length < 2) return;
  const mx = Math.max(...data) * 1.1 || 1;
  g.beginPath(); data.forEach((v, i) => { const x = i / (data.length - 1) * w, y = h - v / mx * h; i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
  g.lineTo(w, h); g.lineTo(0, h); g.closePath(); g.fillStyle = col.replace(/[\d.]+\)$/, '.12)'); g.fill();
}
const ringV = { util: 0, mem: 0 }, ringT = { util: 0, mem: 0 }, ringLbl = { util: '—', mem: '—' };
function drawRing(id, v, lbl, sub, col) {
  const c = $(id), g = c.getContext('2d'), w = c.width, cx = w / 2, r = w * .38;
  g.clearRect(0, 0, w, w); g.lineCap = 'round'; g.lineWidth = w * .06;
  g.strokeStyle = 'rgba(255,255,255,.07)'; g.beginPath(); g.arc(cx, cx, r, .75 * Math.PI, 2.25 * Math.PI); g.stroke();
  g.strokeStyle = v > .9 ? '#f85149' : v > .75 ? '#d29922' : col;
  g.beginPath(); g.arc(cx, cx, r, .75 * Math.PI, (.75 + 1.5 * v) * Math.PI); g.stroke();
  g.fillStyle = '#d5dbe4'; g.font = `600 ${w * .17}px Cascadia Code, Consolas, monospace`; g.textAlign = 'center'; g.fillText(lbl, cx, cx + w * .05);
  g.fillStyle = '#566070'; g.font = `${w * .085}px Segoe UI, sans-serif`; g.fillText(sub, cx, cx + w * .2);
}
let gpuHist = [];
function drawGpuHist() {
  const [g, w, h] = fit($('gpuHist')); g.clearRect(0, 0, w, h);
  if (gpuHist.length < 2) return;
  const tot = S.snap?.gpu?.mem_total || 24;
  const plot = (f, col, fill) => { g.beginPath(); gpuHist.forEach((p, i) => { const x = i / (gpuHist.length - 1) * w, y = h - clamp(f(p)) * (h - 2) - 1; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke(); if (fill) { g.lineTo(w, h); g.lineTo(0, h); g.fillStyle = fill; g.fill(); } };
  plot((p) => p[2] / tot, 'rgba(164,139,250,.9)', 'rgba(164,139,250,.06)');
  plot((p) => p[1] / 100, rgba(.95));
}

/* ---------------------------------------------------------------- main loop */
// Everything that moves is drawn every display frame (owner 11-04: "a high refresh rate so it doesn't feel like a
// choppy laggy application"); the 30 fps canvases and 4 Hz meters looked choppy. It stays cheap by drawing only what
// changed: the orb and spectrum while sound or thinking moves them, the scope when a telemetry frame lands, the
// sparklines and GPU history when a snapshot lands, the rings while they glide, meters as compositor transforms.
const dirty = { scope: true, snap: true };
let idleDrawn = false, lastText = 0;
const ringDrawn = { util: -1, mem: -1, lu: '', lm: '', ac: '' };
function frame(t) {
  requestAnimationFrame(frame);           // first: an exception below must never stop the loop
  if (document.hidden) return;
  const f = S.fast, sm = S.sm;
  sm.in = lerp(sm.in, f.in || 0, .35); sm.clean = lerp(sm.clean, f.clean || 0, .35); sm.out = lerp(sm.out, f.out || 0, .35);
  let specMoving = false;
  for (let i = 0; i < 32; i++) {
    const want = (f.spec && f.spec[i]) || 0, v = lerp(sm.spec[i] || 0, want, want > sm.spec[i] ? .5 : .12);
    if (v > .01) specMoving = true;
    sm.spec[i] = v;
  }
  sm.energy = lerp(sm.energy, clamp(Math.max((db(sm.clean) + 50) / 50, (db(sm.out) + 50) / 50)), .08);
  sm.think = lerp(sm.think, f.thinking ? 1 : 0, .1);
  const themeMoved = tickTheme();
  // Quiet room, nothing thinking, colours settled: the orb and spectrum look the same every frame,
  // so draw them once and stop until something moves.
  const busy = themeMoved || specMoving || sm.think > .02 || db(sm.in) > -60 || db(sm.out) > -60;
  if (busy || !idleDrawn) { drawOrb(t); drawSpec(); idleDrawn = !busy; }
  if (dirty.scope) { dirty.scope = false; drawScope(); }
  meter('mIn', 'vIn', sm.in, t); meter('mClean', 'vClean', sm.clean, t); meter('mOut', 'vOut', sm.out, t);
  if (shown($('rUtil'))) {
    ringV.util = lerp(ringV.util, ringT.util, .12); ringV.mem = lerp(ringV.mem, ringT.mem, .12);
    const ac = `rgb(${accentRGB.map(Math.round).join(',')})`;
    if (Math.abs(ringV.util - ringDrawn.util) > .002 || ringLbl.util !== ringDrawn.lu || ac !== ringDrawn.ac) {
      ringDrawn.util = ringV.util; ringDrawn.lu = ringLbl.util; ringDrawn.ac = ac;
      drawRing('rUtil', ringV.util, ringLbl.util, 'GPU LOAD', ac);
    }
    if (Math.abs(ringV.mem - ringDrawn.mem) > .002 || ringLbl.mem !== ringDrawn.lm) {
      ringDrawn.mem = ringV.mem; ringDrawn.lm = ringLbl.mem;
      drawRing('rMem', ringV.mem, ringLbl.mem, 'VRAM', '#a48bfa');
    }
  }
  if (dirty.snap || themeMoved) {
    dirty.snap = false;
    if (shown($('sE2E'))) { spark('sE2E', series.e2e, 'rgba(52,211,153,.9)'); spark('sHdr', series.hdr, rgba(.9)); spark('sPause', series.pause, 'rgba(251,191,36,.9)'); }
    if (shown($('gpuHist'))) drawGpuHist();
  }
  if (t - lastText > 100) { lastText = t; setText($('curSpk'), S.typingUserSpk ? `speaker ${S.typingUserSpk}` : '—'); }
  flushFeeds();
}

/* ---------------------------------------------------------------- boot */
applyTheme(); renderPersonaSelects(); renderPersonaCards(); renderChat();
connectTel(); connectWs();
requestAnimationFrame(frame);

// ---- Eyes (screen tool owner switch + last look) ----
let _eyeTs = 0;
function renderEyes(sc) {
  const el = $('pEyes'); if (!el) return;
  if (!sc) { el.className = 'pill eyes'; return; }
  el.className = 'pill eyes ' + (sc.enabled ? 'ok' : 'warn');
  el.childNodes[1].textContent = sc.enabled ? 'Eyes' : 'Eyes off';
  const meta = $('eyeMeta'), img = $('eyeImg');
  const live = sc.live;
  if (live && live.on) {             // Live screen: the newest live frame, refreshed every 2 s while the popup is open
    const now = Math.floor(Date.now() / 2000);
    // only while someone hovers the pill: a hidden preview still cost a frame grab and a JPEG decode every 2 s
    if (live.state === 'capturing' && now !== _eyeTs && el.matches(':hover')) { _eyeTs = now; img.src = '/api/screen/last?live=' + now; img.classList.add('on'); }
    const state = live.state === 'capturing' ? 'watching live' : `can't see it (${live.state})`;
    meta.textContent = `Live screen: ${live.watching}${live.title ? ' · ' + live.title : ''} · ${state} · looked ${sc.count}x`
      + (live.remembered ? ` · remembers ${live.remembered} things` : '');
    meta.title = live.remembered ? 'Screen log: /api/screen/log (owner only). Say "forget what you saw" to drop the last two minutes.' : '';
    return;
  }
  if (sc.last_ts) {
    if (sc.last_ts !== _eyeTs) {
      if (_eyeTs) { el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }
      _eyeTs = sc.last_ts; img.src = '/api/screen/last?t=' + sc.last_ts; img.classList.add('on');
    }
    const ago = Math.max(0, Math.round(Date.now() / 1000 - sc.last_ts));
    meta.textContent = `Looked ${sc.count}x · last ${ago < 60 ? ago + 's' : Math.round(ago / 60) + 'm'} ago` + (sc.last_reason ? ` · "${sc.last_reason}"` : '') + (sc.enabled ? '' : ' · switched OFF');
  } else meta.textContent = sc.enabled ? 'On. Agents can look when someone asks.' : "Off. Agents will say they can't see.";
}
// ---- Brain (LLM backend: local Ollama <-> remote vLLM; hidden if not configured) ----
let _brainInfo = null;
function renderBrain(tb) {
  const el = $('pBrain'); if (!el) return;
  if (!_brainInfo || !_brainInfo.available) { el.classList.add('hidden'); return; }
  const b = tb || {};
  const remote = b.mode === 'remote', busy = b.switching || S.brainBusy;
  const r = b.remote || {};
  el.className = 'pill brain ' + (busy ? 'busy' : remote ? (r.healthy === false ? 'warn' : 'ok') : '');
  $('brainLbl').textContent = busy ? 'Switching…' : remote ? 'Remote LLM' : 'Local LLM';
  const lines = [];
  lines.push(remote ? `<b>${esc(_brainInfo.remote_label || 'Remote')}</b>` : `<b>Local</b> · ${esc(_brainInfo.local_model || '')}`);
  if (remote) {
    lines.push(`${r.requests || 0} requests · ${r.fallbacks || 0} fell back to local`);
    if (r.healthy === false) lines.push('Unreachable right now, using local until it recovers.');
    lines.push('Local model unloaded (VRAM free for STT + TTS).');
  } else lines.push(`Remote available: ${esc(_brainInfo.remote_label || '')}`);
  if (S.brainErr) lines.push(`<span style="color:var(--bad)">${esc(S.brainErr)}</span>`);
  lines.push(busy ? 'Switching…' : `Click to switch to ${remote ? 'LOCAL' : 'REMOTE'}.`);
  $('brainMeta').innerHTML = lines.join('<br>');
}
async function loadBrain() {
  try { _brainInfo = await (await fetch('/api/llm_backend')).json(); } catch (_) { _brainInfo = null; }
  renderBrain(_brainInfo ? { mode: _brainInfo.mode, switching: _brainInfo.busy, remote: _brainInfo.remote } : null);
}
(function wireBrain() {
  const el = $('pBrain'); if (!el) return;
  loadBrain(); setInterval(loadBrain, 15000);
  el.addEventListener('click', async (e) => {
    if (e.target.closest('.eyepop') || S.brainBusy || !_brainInfo) return;
    const want = (S.snap?.llm_backend?.mode || _brainInfo.mode) === 'remote' ? 'local' : 'remote';
    S.brainBusy = true; S.brainErr = ''; renderBrain(S.snap?.llm_backend);
    try {
      const r = await fetch('/api/llm_backend', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: want }) });
      const st = await r.json(); if (st.error) S.brainErr = st.error; _brainInfo = st;
    } catch (err) { S.brainErr = String(err); }
    S.brainBusy = false; renderBrain(_brainInfo ? { mode: _brainInfo.mode, remote: _brainInfo.remote } : null);
  });
})();
// ---- Owner controls: hard mute, quiet mode, talkativeness, muted speakers ----
async function ownerPost(body) {
  // -> {ok, status, body}; an error answer is shown as a toast and never replaces the owner state
  try {
    const r = await fetch('/api/owner', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    let st = null; try { st = await r.json(); } catch (_) { /* not JSON */ }
    if (!r.ok || !st) { toast((st && st.error) || `owner setting failed (${r.status})`, '#f87171'); return { ok: false, status: r.status, body: st }; }
    S.snap = S.snap || {}; S.snap.owner = st; renderOwner(st); renderDock();
    return { ok: true, status: r.status, body: st };
  } catch (_) { toast("the backend didn't answer", '#f87171'); return { ok: false, status: 0, body: null }; }
}
// ---- owner voice recognition (owner_voice.py): the owner's voice is recognized in every new call once they press
// "This is me" on their roster row; "Not me" undoes a wrong recognition for this call; "Forget my voice" deletes it.
const ownerVoice = () => (S.snap && S.snap.owner && S.snap.owner.voice) || null;
function watchVoice(v) {
  // a label that appears in voice.labels between two snapshots was recognized by voice (not on the first snapshot,
  // and not the label this dashboard just enrolled)
  const labels = (v && v.labels) || [];
  if (S.voiceLabels !== undefined && v)
    for (const l of labels) if (!S.voiceLabels.includes(l) && l !== S.justEnrolled) voiceNotice(`Recognized ${v.name} by voice`, l);
  if (S.justEnrolled && labels.includes(S.justEnrolled)) S.justEnrolled = null;
  S.voiceLabels = labels;
}
let voiceNoteT = 0;
function voiceNotice(text, label) {
  const n = $('voiceNote'), b = $('voiceNoteNotMe');
  n.querySelector('.vtext').textContent = text;
  b.hidden = !label; b.dataset.notme = label || '';
  n.hidden = false; clearTimeout(voiceNoteT); voiceNoteT = setTimeout(() => { n.hidden = true; }, label ? 12000 : 7000);
}
function closeMeForm() { const f = $('meForm'); f.hidden = true; f.innerHTML = ''; S.meLabel = null; }
function openMeForm(label) {
  const f = $('meForm'), v = ownerVoice();
  S.meLabel = label;
  f.innerHTML = `<label>Your name for this voice <input id="meName" maxlength="40" autocomplete="off" value="${esc((v && v.name) || '')}"></label>`
    + '<button class="btn primary" id="meSave">Save</button><button class="btn" id="meCancel">Cancel</button>';
  f.hidden = false;
  const inp = $('meName'); inp.focus(); inp.select();
  inp.onkeydown = (e) => { if (e.key === 'Enter') saveMe(); else if (e.key === 'Escape') closeMeForm(); };
  $('meSave').onclick = saveMe; $('meCancel').onclick = closeMeForm;
}
async function saveMe() {
  const label = S.meLabel, name = ($('meName') || {}).value || '';
  if (!label) return;
  S.justEnrolled = label;
  const r = await ownerPost({ owner_voice: { label, name: name.trim() } });
  if (r.ok) { closeMeForm(); voiceNotice('Saved. Atlas will recognize your voice after a few lines in each new call. Saving again in another call makes it steadier.', null); }
  else S.justEnrolled = null;
}
function openForget() {
  const f = $('meForm');
  S.meLabel = null;
  f.innerHTML = '<span class="msg">Delete your saved voice? (Voice data already in call recordings is not deleted.)</span>'
    + '<button class="btn danger" id="meDelete">Delete</button><button class="btn" id="meCancel">Cancel</button>';
  f.hidden = false;
  $('meCancel').onclick = closeMeForm;
  $('meDelete').onclick = async () => { const r = await ownerPost({ owner_voice: null }); if (r.ok) { closeMeForm(); toast('saved voice deleted'); } };
  $('meDelete').focus();
}
// a name cached from a chat line stops showing once its label is no longer recognized as the owner (chat backlog
// replays would otherwise bring the owner's name back onto a friend's row)
function cachedName(id) {
  const n = _spkNames[id], v = ownerVoice();
  if (n && v && v.name && n === v.name && !(v.labels || []).includes(id)) return '';
  return n || '';
}
function renderOwner(o) {
  if (!o) return;
  watchVoice(o.voice);
  const m = $('pMute'), q = $('pQuiet'); if (!m || !q) return;
  m.classList.toggle('on', !!o.mute); $('muteLbl').textContent = o.mute ? 'Muted' : 'Live';
  q.classList.toggle('on', o.quiet_s > 0);
  $('quietLbl').textContent = o.quiet_s > 0 ? `Quiet ${Math.ceil(o.quiet_s / 60)}m` : 'Quiet';
  const sl = $('talkSlider'), pid = S.persona || o.active;
  const v = (o.talkativeness || {})[pid] ?? o.active_talkativeness;
  if (sl && document.activeElement !== sl && v != null) { sl.value = v; $('talkVal').textContent = Number(v).toFixed(2); }
}
(function wireOwner() {
  const m = $('pMute'), q = $('pQuiet'), sl = $('talkSlider'); if (!m || !q) return;
  m.addEventListener('click', (e) => { if (!e.target.closest('.eyepop')) ownerPost({ mute: 'toggle' }); });
  q.addEventListener('click', (e) => { if (!e.target.closest('.eyepop')) ownerPost({ quiet: 'toggle' }); });
  for (const el of [m, q]) el.addEventListener('keydown', (e) => {   // keyboard: Enter or Space toggles, like a button
    if ((e.key === 'Enter' || e.key === ' ') && e.target === el) { e.preventDefault(); el.click(); } });
  if (sl) {
    sl.addEventListener('input', () => { $('talkVal').textContent = Number(sl.value).toFixed(2); });
    sl.addEventListener('change', () => ownerPost({ talkativeness: { agent: S.persona || S.snap?.owner?.active, value: Number(sl.value) } }));
  }
  document.addEventListener('click', (e) => {
    const me = e.target.closest('[data-me]');
    if (me) { openMeForm(me.dataset.me); return; }
    const nm = e.target.closest('[data-notme]');
    if (nm && nm.dataset.notme) { if (nm.id === 'voiceNoteNotMe') $('voiceNote').hidden = true; ownerPost({ owner_voice: { not_me: nm.dataset.notme } }); return; }
    if (e.target.closest('[data-forgetme]')) { openForget(); return; }
    const b = e.target.closest('[data-mute]'); if (!b) return;
    const lab = b.getAttribute('data-mute');
    ownerPost(b.classList.contains('on') ? { unmute_speaker: lab } : { mute_speaker: lab });
  });
  fetch('/api/owner').then((r) => r.json()).then((o) => { S.snap = S.snap || {}; S.snap.owner = S.snap.owner || o; renderOwner(o); renderDock(); }).catch(() => {});
})();
(function wireEyes() {
  const el = $('pEyes'); if (!el) return;
  el.addEventListener('mouseenter', () => { _eyeTs = 0; if (S.snap) renderEyes(S.snap.screen); });   // fresh frame on open
  el.addEventListener('click', async (e) => {
    if (e.target.closest('.eyepop')) return;
    const on = !(S.snap && S.snap.screen && S.snap.screen.enabled);
    try { const r = await fetch('/api/screen', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: on }) });
          const st = await r.json(); if (S.snap) S.snap.screen = st; renderEyes(st); } catch (_) {}
  });
})();

// Speaker label: learned name, else 'Guest N' -- raw diarizer tags (S1) never shown.
const _spkNames = {};
function spkLabel(m) {
  const id = m.speakerId;
  if (!id || id === 'you') return 'You';
  if (m.speakerName) _spkNames[id] = m.speakerName;
  const known = cachedName(id);
  if (known) return known;
  const n = /^S(\d+)$/.exec(id);
  return n ? 'Guest ' + n[1] : id;
}
// ---- guide strip: Set up -> Make an agent -> Join a call -> Review (journey.py)
const guideEl = document.getElementById('guide');
async function refreshGuide(action) {
  try {
    const opt = action ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }) } : {};
    const r = await fetch('/api/journey', opt); if (!r.ok) return;
    const j = await r.json();
    // First-run help only: hidden for good once there's a first conversation.
    guideEl.classList.toggle('hidden', !j.show);
    if (!j.show) return;
    const cur = j.steps.find(s => s.id === j.current);
    document.getElementById('guideTitle').textContent = 'Next: ' + (cur ? cur.label : '');
    document.getElementById('guideHint').textContent = j.hint;
  } catch (e) { /* backend down: leave the strip as it was */ }
}
document.getElementById('guideHide').onclick = () => refreshGuide('hide');
refreshGuide(); setInterval(refreshGuide, 30000);

/* ---------------------------------------------------------------- collapsible sections */
(() => {
  // v2: panels are grouped (Call, System) and the defaults changed, so they apply once (design D5)
  const KEY = 'atlas.collapsed.v2';
  let saved; try { saved = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch { saved = null; }
  // first run: the diagnostics (System) start folded except the Pipeline feed; the Call panels start open
  const DEFAULT = ['Model context', 'GPU', 'Models', 'Audio routing'];
  const name = (p) => (p.querySelector('.ptitle')?.childNodes[0]?.textContent || '').trim();
  const closed = new Set(saved || DEFAULT);
  document.querySelectorAll('.side .panel').forEach((p) => {
    if (closed.has(name(p))) p.classList.add('collapsed');
    p.querySelector('.ptitle').addEventListener('click', (e) => {
      if (e.target.closest('.tabs')) return;
      p.classList.toggle('collapsed');
      if (!p.classList.contains('collapsed')) { if (S.snap) renderSnap(S.snap); dirty.snap = true; logRebuild = true; ringDrawn.util = ringDrawn.mem = -1; }
      const now = [...document.querySelectorAll('.side .panel.collapsed')].map(name);
      try { localStorage.setItem(KEY, JSON.stringify(now)); } catch {}
    });
  });
})();
