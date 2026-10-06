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
  voices: [], persona: '', voice: '', switching: null,
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
  if (k === _themeKey) return;           // converged: no style writes, no restyle of the page
  _themeKey = k;
  document.documentElement.style.setProperty('--accent', `rgb(${k})`);
  document.documentElement.style.setProperty('--accent-rgb', k);
}
const rgba = (a) => `rgba(${accentRGB.map(Math.round).join(',')},${a})`;
const SPK_COLORS = ['#5eead4', '#fbbf24', '#f472b6', '#60a5fa', '#a3e635', '#fb7185', '#c084fc', '#38bdf8'];
function spkColor(label) { if (!label || label === 'you' || label === 'YOU') return '#5eead4'; let h = 0; for (const c of label) h = (h * 31 + c.charCodeAt(0)) >>> 0; return SPK_COLORS[h % SPK_COLORS.length]; }

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
  vs.onchange = () => { S.voice = vs.value; send({ type: 'set_voice', voice: vs.value }); toast(`voice → ${vs.value}`); };
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
      d.onclick = (e) => (e.shiftKey ? panelClick(p) : avClick(p));
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
    el.classList.toggle('panel', (S.panel?.members || []).includes(id));
  }
}
// Agent panel (owner 10-06): Shift+click another agent to bring it on as co-host;
// Shift+click a panel member to end the panel. One agent speaks per turn.
async function refreshPanel() {
  try { S.panel = await (await fetch('/api/panel')).json(); renderDock(); } catch (_) {}
}
async function panelClick(p) {
  const members = S.panel?.members || [];
  const end = members.includes(p.id) || p.id === S.persona;
  const body = { members: end ? [] : [S.persona, p.id] };
  try {
    const r = await fetch('/api/panel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) return toast(j.error || 'could not change the panel');
    S.panel = j; renderDock(); hideTip();
    const nm = (id) => ((S.personas.find((x) => x.id === id) || {}).name || id).toUpperCase();
    toast(j.active ? `PANEL · ${j.members.map(nm).join(' + ')}` : 'panel ended', p.accent);
  } catch (_) { toast('could not change the panel'); }
}
refreshPanel(); setInterval(refreshPanel, 5000);
// Background mode (owner 10-06): click the ACTIVE agent to toggle it. The agent then
// only answers when spoken to; its name opens a short direct exchange.
const isBackground = (id) => ((S.snap?.owner?.background) || []).includes(id);
async function avClick(p) {
  if (p.id !== S.persona) return setPersona(p.id);
  const on = !isBackground(p.id);
  const st = await ownerPost({ background: { agent: p.id, on } });
  if (!st) return toast('could not change background mode');
  renderDock(); hideTip();
  toast(on ? `${(p.name || p.id).toUpperCase()} · background mode — answers only when spoken to`
           : `${(p.name || p.id).toUpperCase()} · conversation mode`, p.accent);
}
function showTip(el, p) {
  const t = $('avTip'); const r = el.getBoundingClientRect();
  const bgm = p.id && isBackground(p.id);
  const st = p.id ? (S.switching === p.id ? 'switching…' : p.id === S.persona ? (S.fast.speaking ? 'speaking' : (bgm ? 'background mode · click for conversation' : 'active · click for background mode')) : (bgm ? 'click to switch · background' : 'click to switch')) : '';
  t.innerHTML = `<div class="n" style="color:${esc(p.accent || 'var(--accent)')}">${esc((p.name || '').toUpperCase())}</div>`
    + `<div class="r">${esc(p.role || '')}</div>`
    + (p.voice || st ? `<div class="m">${p.voice ? 'voice · ' + esc(p.voice) : ''}${p.voice && st ? ' — ' : ''}${esc(st)}${p.id ? ' · right-click: memory · shift-click: panel' : ''}</div>` : '');
  t.style.left = Math.max(8, Math.min(innerWidth - 250, r.left + r.width / 2 - 80)) + 'px';
  t.style.top = (r.bottom + 10) + 'px';
  t.classList.add('show');
}
function hideTip() { $('avTip').classList.remove('show'); }
// Old call sites keep working.
function renderPersonaSelects() { renderVoiceSelect(); renderDock(); }
function renderPersonaCards() { renderDock(); }
function setPersona(id) {
  if (!id || id === S.persona) return renderDock();
  hideTip();
  S.switching = id; S.persona = id; applyTheme();
  send({ type: 'set_persona', persona: id });
  renderPersonaSelects();
  S.chat.push({ role: 'sys', content: `— switched to ${persona(id).name.toUpperCase()} —` }); renderChat();
  toast(`persona → ${persona(id).name.toUpperCase()}`, persona(id).accent);
}
async function refreshPersonas() {
  try {
    const r = await fetch('/api/personas'); const m = await r.json();
    applyPersonasMsg(m);
  } catch (e) { /* ws list_personas will cover it */ }
}
function applyPersonasMsg(m) {
  if (m.personas) S.personas = m.personas.map((p) => ({ id: p.id, name: p.name, role: p.role, accent: p.accent, voice: p.voice, custom: !!p.custom }));
  if (m.voices) S.voices = m.voices;
  if (m.current) { S.persona = m.current.persona; S.voice = m.current.voice; }
  S.switching = null; applyTheme(); renderPersonaSelects();
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
function disarmWipe() { clearTimeout(MEM.armT); const b = $('memWipe'); b.classList.remove('arm'); b.textContent = 'DELETE MEMORIES'; }
async function loadMemCounts() {
  const box = $('memCounts'); const p = MEM.p; if (!p) return;
  try {
    const r = await fetch(`/api/memory/${encodeURIComponent(p.id)}`); const m = await r.json();
    if (!r.ok) throw new Error(m.error || r.status);
    if (MEM.p !== p) return;
    box.innerHTML = `<span><b>${m.approved.length}</b> approved</span><span><b>${m.candidates.length}</b> waiting for review</span><span><b>${m.learned.length}</b> learned in calls</span>`;
  } catch (e) { box.innerHTML = `<span>memory unavailable: ${esc(String(e.message || e))}</span>`; }
}
function openMemory(p) {
  MEM.p = p;
  $('memTitle').textContent = `${(p.name || p.id).toUpperCase()} · MEMORY`;
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
    b.classList.add('arm'); b.textContent = 'CLICK AGAIN TO DELETE';
    MEM.armT = setTimeout(disarmWipe, 4000); return;
  }
  disarmWipe(); b.disabled = true;
  const w = $('memWindow').value;
  try {
    const r = await fetch(`/api/memory/${encodeURIComponent(p.id)}/wipe`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ window: w }) });
    const m = await r.json();
    if (!r.ok) throw new Error(m.error || r.status);
    const n = (m.items || 0) + (m.learned || 0);
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
  $('crName2').textContent = (d.name || $('crName').value).toUpperCase();
  $('crSrc').textContent = d.source === 'llm' ? 'DRAFTED BY MODEL' : d.source === 'fallback' ? 'MODEL FAILED — EDIT BY HAND' : d.source === 'heart' ? 'FROM HEART.MD' : 'MANUAL';
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
    el.innerHTML = `<div class="who"><span class="t">${m.time || ''}</span>${esc(p.name.toUpperCase())}</div><div class="body"></div>`;
  } else {
    el.style.setProperty('--sc', spkColor(m.label));
    el.innerHTML = `<div class="who">${esc(m.label || 'YOU')}<span class="t">${m.time || ''}</span></div><div class="body"></div>`;
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
$('clearView').onclick = () => { S.chat = []; renderChat(); };
$('clearBtn').onclick = () => { send({ type: 'clear_history' }); S.chat.push({ role: 'sys', content: '— model history cleared —' }); renderChat(); toast('history cleared'); };
const speedWord = (v) => (v <= 10 ? 'SNAPPY' : v <= 40 ? 'QUICK' : v <= 70 ? 'RELAXED' : 'PATIENT');
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
    case 'partial_user_request': S.typingUser = c; S.typingUserSpk = m.speakerId || S.typingUserSpk; renderChat(); break;
    case 'final_user_request':
      if (c.trim()) S.chat.push({ role: 'user', content: c, label: spkLabel(m), time: hhmm() });
      if (guideEl && c.trim()) guideEl.classList.add('hidden'); // a live conversation needs no 'next step' card
      S.typingUser = ''; S.typingUserSpk = ''; renderChat(); break;
    case 'partial_assistant_answer': S.typingAssistant = c; renderChat(); break;
    case 'final_assistant_answer':
      if (c.trim()) S.chat.push({ role: 'assistant', content: c, persona: m.personaId || S.persona, time: hhmm() });
      S.typingAssistant = ''; renderChat(); break;
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
const pushDevices = () => { send({ type: 'set_audio_devices', input: +$('devIn').value, output: +$('devOut').value }); toast('audio routing updated'); };
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
    S.scope.push([f.in || 0, f.out || 0]); if (S.scope.length > 180) S.scope.shift();
    updatePartial(); updateLamps(); return;
  }
  if (f.type === 'snap') { S.snap = f; S.bridge = !!f.bridge; renderSnap(f); if (S.wsOpen) maybeStartBrowserAudio(); return; }
  if (f.type === 'events') { f.items.forEach(onEvent); return; }
  if (f.type === 'lines') { appendLogs(f.items); }
}

/* ---------------------------------------------------------------- core column */
const db = (x) => (x > 1e-6 ? 20 * Math.log10(x) : -99);
function meter(id, vid, x) { const d = db(x); $(id).style.width = `${clamp((d + 60) / 60) * 100}%`; $(vid).textContent = d < -60 ? 'idle' : `${d.toFixed(0)} dB`; $(vid).classList.toggle('idle', d < -60); }
function updatePartial() {
  const t = S.fast.partial || S.typingUser;
  const el = $('partialTxt');
  if (t && (S.fast.recording || S.typingUser)) { el.textContent = t; el.classList.remove('empty'); }
  else if (!S.fast.recording) { el.textContent = 'waiting for speech…'; el.classList.add('empty'); }
}
function updateLamps() {
  const f = S.fast, fl = S.snap?.pipeline?.floor;
  $('lHear').classList.toggle('on', !!f.recording);
  $('lThink').classList.toggle('on', !!f.thinking);
  $('lSpeak').classList.toggle('on', !!f.speaking);
  $('lQuiet').classList.toggle('on', !!fl?.quiet);
  const st = f.speaking ? 'SPEAKING' : f.thinking ? 'THINKING' : f.recording ? 'LISTENING' : fl?.quiet ? 'QUIET MODE' : S.telOpen ? 'READY' : 'OFFLINE';
  $('orbState').textContent = st;
  // pipeline stepper
  const now = performance.now();
  const gen = S.snap?.pipeline?.gen;
  let stage = null, done = [];
  if (f.speaking) { stage = 'speak'; done = ['hear', 'end', 'decide', 'gen']; }
  else if (f.thinking && gen?.decision === 'SPEAK') { stage = 'gen'; done = ['hear', 'end', 'decide']; }
  else if (f.thinking) { stage = 'decide'; done = f.recording ? ['hear'] : ['hear', 'end']; }
  else if (now - S.turnEndAt < 900) { stage = 'end'; done = ['hear']; }
  else if (f.recording) stage = 'hear';
  document.querySelectorAll('#pipe div').forEach((d) => { d.classList.toggle('on', d.dataset.s === stage); d.classList.toggle('done', done.includes(d.dataset.s)); });
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
    if (!po) { vs.textContent = '—'; vs.style.color = ''; }
    else {
      const last = po.last_reply_underruns || 0;
      vs.textContent = `${po.underruns} gaps / ${po.replies} replies · ${Math.round(po.silence_ms)}ms · buf ${po.prebuffer_ms}ms`;
      vs.style.color = last ? 'rgb(251,191,36)' : (po.underruns ? '' : 'rgb(52,211,153)');
    }
  }
  $('pBridge').lastChild.textContent = s.bridge ? 'CALL' : 'BROWSER';
  renderEyes(s.screen);
  renderOwner(s.owner);
  renderBrain(s.llm_backend);
  const u = Math.floor(s.uptime || 0); $('uptime').textContent = `${String(Math.floor(u / 3600)).padStart(2, '0')}:${String(Math.floor(u / 60) % 60).padStart(2, '0')}:${String(u % 60).padStart(2, '0')}`;
  const pl = s.pipeline || {};
  $('modelName').textContent = (s.llm_backend?.mode === 'remote' && s.llm_backend.remote) ? `${s.llm_backend.remote.model} @ remote` : (pl.model || '—');
  if (pl.persona && !S.switching && pl.persona !== S.persona) { S.persona = pl.persona; applyTheme(); renderPersonaSelects(); }
  if (S.switching && pl.persona === S.switching && pl.voice) { S.switching = null; S.voice = pl.voice; renderPersonaSelects(); }
  renderPersonaCards();

  // latency
  const L = s.latency || {};
  $('kE2E').innerHTML = L.e2e_s?.n ? fmt(L.e2e_s.p50, 's', 2) : '—';
  $('kHdr').innerHTML = L.header_ms?.n ? fmt(L.header_ms.p50, 'ms') : '—';
  $('kPause').innerHTML = L.pause_s?.n ? fmt(L.pause_s.p50, 's', 2) : '—';
  series.e2e = L.e2e_s?.series || []; series.hdr = L.header_ms?.series || []; series.pause = L.pause_s?.series || [];
  const C = s.counters || {};
  $('cSpeak').textContent = C.decision_SPEAK || 0; $('cHold').textContent = C.decision_HOLD || 0;
  $('cBad').textContent = (C.decision_INVALID || 0) + (C.error || 0);
  $('turnsN').textContent = `${L.e2e_s?.n || 0} replies · p90 ${L.e2e_s?.n ? L.e2e_s.p90.toFixed(2) + 's' : '—'}`;

  // gpu
  const g = s.gpu || {};
  if (!g.error) {
    ringT.util = (g.util || 0) / 100; ringT.mem = (g.mem_used || 0) / (g.mem_total || 1);
    ringLbl.util = `${g.util ?? 0}%`; ringLbl.mem = `${(g.mem_used || 0).toFixed(1)}G`;
    $('gpuName').textContent = (g.name || '').replace('NVIDIA GeForce ', '');
    $('gpuStats').innerHTML = `temp <b>${g.temp}°C</b><br>power <b>${(g.power || 0).toFixed(0)}</b>/${(g.power_limit || 0).toFixed(0)} W<br>clock <b>${g.clock}</b> MHz<br>free <b>${((g.mem_total || 0) - (g.mem_used || 0)).toFixed(1)}</b> GB`;
    $('gpuProcs').innerHTML = (g.procs || []).map((p) => `<div class="rl-item"><span class="n">${esc(p.name)}</span><span class="m">${p.gb.toFixed(2)} GB</span><div class="minibar"><b style="width:${clamp(p.gb / (g.mem_total || 24)) * 100}%"></b></div></div>`).join('');
  } else { $('gpuStats').textContent = g.error; }
  gpuHist = s.gpu_hist || [];

  // models + process
  const ms = s.ollama?.models || [];
  $('models').innerHTML = ms.length ? ms.map((m) => {
    const exp = m.expires ? Math.max(0, (Date.parse(m.expires) - Date.now()) / 60000) : null;
    return `<div class="rl-item"><span class="n">${esc(m.name)}</span><span class="m">${m.vram_gb.toFixed(1)} GB · ${esc(m.params || '')} ${esc(m.quant || '')}</span><div class="minibar"><b style="width:${exp == null ? 0 : clamp(exp / 30) * 100}%"></b></div><span class="m" style="grid-column:1/-1">ctx ${m.ctx ?? '—'} · keep-alive ${exp == null ? '—' : exp.toFixed(0) + ' min'}</span></div>`;
  }).join('') : `<div class="stats">${s.ollama?.up ? 'no model resident' : 'ollama unreachable'}</div>`;
  const p = s.proc || {};
  $('procStats').textContent = p.error ? '' : `cpu ${(p.sys_cpu || 0).toFixed(0)}% · ram ${(p.rss_gb || 0).toFixed(1)}G · ${p.threads} thr`;

  // room
  const fl = pl.floor || {};
  $('floorRow').innerHTML = [
    `<span class="chip ${fl.partner ? 'on' : ''}">partner ${esc((pl.names || {})[fl.partner] || _spkNames[fl.partner] || fl.partner || 'none')}</span>`,
    `<span class="chip ${fl.quiet ? 'q' : ''}">${fl.quiet ? `quiet ${Math.ceil(fl.quiet_left)}s` : 'open floor'}</span>`,
    `<span class="chip">room ${fl.room_size ?? 0}</span>`,
    `<span class="chip">history ${pl.history_len ?? 0}</span>`,
  ].join('');
  const roster = pl.roster || [];
  const dz = Object.fromEntries((s.speakers || []).map((x) => [x.label, x]));
  const liveSpk = S.typingUserSpk;
  $('roomN').textContent = `${roster.length} people · ${(s.speakers || []).length} voices`;
  $('roster').innerHTML = roster.length ? roster.map((r) => {
    const col = spkColor(r.id), d = dz[r.id];
    const nm = r.name || _spkNames[r.id] || '';
    const tags = [r.addressed ? `<span class="tag hot">called ${r.addressed}×</span>` : ''].join('');
    const muted = (s.owner?.muted_speakers || []).includes(String(r.id).toUpperCase());
    const ago = r.idle_s < 60 ? r.idle_s.toFixed(0) + 's' : Math.round(r.idle_s / 60) + 'm';
    const meta = `${r.turns} turns${d ? ` · ${d.speech_s}s` : ''} · ${ago}`;
    return `<div class="spk ${r.id === liveSpk ? 'live' : ''} ${r.id === fl.partner ? 'partner' : ''} ${muted ? 'muted' : ''}" title="${esc(r.id)} · ${r.words} words">
      <div class="sav" style="--c:${col}">${esc((nm || r.id).slice(0, 2))}</div>
      <div class="sbody"><div class="sname"><span>${esc(nm || 'Unknown voice')}</span>${tags}<span class="smeta">${meta}</span></div><div class="line">${esc(r.last)}</div></div>
      <span class="mutebtn ${muted ? 'on' : ''}" data-mute="${esc(r.id)}" title="${muted ? 'Unmute' : 'Ignore this speaker (agent never answers them)'}">${muted ? 'muted' : 'mute'}</span></div>`;
  }).join('') : `<div class="stats">${(s.speakers || []).map((x) => `${esc(x.label)} · ${x.speech_s}s`).join('<br>') || 'nobody has spoken yet'}</div>`;

  // threads
  const th = pl.threads || {};
  const open = th.open || [];
  $('threads').innerHTML = th.error ? `<div class="stats">${esc(th.error)}</div>` : (open.length ? open.map((t) => {
    const act = th.active && th.active.speaker === t.speaker && th.active.text === t.text;
    return `<div class="thread ${act ? 'active' : ''}"><div class="top"><span style="color:${spkColor(t.speaker)}">${esc(t.speaker)}${act ? ' · ANSWERING' : ''}</span><span>score ${t.score}</span></div><div>${esc(t.text)}</div><div class="rs">${(t.reasons || []).map((x) => `<span>${esc(x)}</span>`).join('')}</div></div>`;
  }).join('') : '<div class="stats">no open threads</div>');

  // context
  const cv = pl.convo || {};
  $('ctxStats').textContent = Object.keys(cv).length ? Object.entries(cv).slice(0, 4).map(([k, v]) => `${k} ${typeof v === 'number' ? Math.round(v) : v}`).join(' · ') : '—';
  $('roomNote').textContent = pl.room_note || '—';
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
    case 'persona': x = e.name; S.switching = null; break;
    case 'voice': x = e.name; break;
    case 'error': x = e.text; toast('pipeline error — see feed', 'var(--bad)'); break;
    case 'speaker': case 'turn_end': case 'tts_start': return;
    default: x = JSON.stringify(e).slice(0, 120);
  }
  const d = document.createElement('div');
  d.className = `ev ${cls}`; d.innerHTML = `<span class="tm">${hhmm(e.t)}</span><span class="k">${esc(k)}</span><span class="x" title="${esc(x)}">${esc(x)}</span>`;
  evBox.prepend(d); while (evBox.children.length > 120) evBox.lastChild.remove();
}
function appendLogs(items) {
  items.forEach((l) => { const d = document.createElement('div'); d.className = l.lvl; d.textContent = `${hhmm(l.t)} ${l.src.padEnd(12)} ${l.msg}`; logBox.appendChild(d); });
  while (logBox.children.length > 300) logBox.firstChild.remove();
  if (S.evTab === 'logs') logBox.scrollTop = logBox.scrollHeight;
}
document.querySelectorAll('.tabs span').forEach((t) => t.onclick = () => {
  S.evTab = t.dataset.t; document.querySelectorAll('.tabs span').forEach((x) => x.classList.toggle('on', x === t));
  evBox.style.display = S.evTab === 'events' ? '' : 'none'; logBox.style.display = S.evTab === 'logs' ? '' : 'none';
  logBox.scrollTop = logBox.scrollHeight;
});

/* ---------------------------------------------------------------- canvases */
function fit(c) { const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr)); if (c.width !== w || c.height !== h) { c.width = w; c.height = h; } return [c.getContext('2d'), w, h, dpr]; }

// orb: thin status ring. Spectrum ticks (call in), output-level arc, thinking sweep.
function drawOrb(t) {
  const [g, w, h, dpr] = fit($('orb'));
  g.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h * .5, R = Math.min(w, h) * .36, sm = S.sm;
  const out = clamp((db(sm.out) + 55) / 50), inn = clamp((db(sm.clean) + 55) / 50);
  g.lineCap = 'round';
  g.lineWidth = 1 * dpr; g.strokeStyle = 'rgba(255,255,255,.07)';
  g.beginPath(); g.arc(cx, cy, R, 0, 7); g.stroke();
  // spectrum ticks around the ring
  const spec = sm.spec, n = spec.length * 2;
  g.lineWidth = 1.6 * dpr;
  for (let i = 0; i < n; i++) {
    const v = spec[i < spec.length ? i : n - 1 - i] || 0;
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    const r0 = R + 5 * dpr, r1 = r0 + 2 * dpr + v * R * .32 * (0.4 + inn);
    g.strokeStyle = `rgba(121,192,255,${.18 + v * .7})`;
    g.beginPath(); g.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0); g.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1); g.stroke();
  }
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
  for (let i = 0; i < n; i++) {
    const v = s[i], bh = Math.max(2, v * h);
    g.fillStyle = `rgba(121,192,255,${.25 + v * .6})`; g.fillRect(i * bw + 1, h - bh, bw - 2, bh);
  }
}
function drawScope() {
  const [g, w, h] = fit($('scope')); g.clearRect(0, 0, w, h);
  const d = S.scope, n = d.length; if (n < 2) return;
  const line = (idx, col) => {
    g.beginPath();
    d.forEach((p, i) => { const x = (i / 179) * w, y = h - clamp((db(p[idx]) + 60) / 60) * (h - 2) - 1; i ? g.lineTo(x, y) : g.moveTo(x, y); });
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
let lastT = 0, lastSlow = 0;
function frame(t) {
  const f = S.fast, sm = S.sm;
  sm.in = lerp(sm.in, f.in || 0, .35); sm.clean = lerp(sm.clean, f.clean || 0, .35); sm.out = lerp(sm.out, f.out || 0, .35);
  for (let i = 0; i < 32; i++) sm.spec[i] = lerp(sm.spec[i] || 0, (f.spec && f.spec[i]) || 0, f.spec && f.spec[i] > sm.spec[i] ? .5 : .12);
  sm.energy = lerp(sm.energy, clamp(Math.max((db(sm.clean) + 50) / 50, (db(sm.out) + 50) / 50)), .08);
  sm.think = lerp(sm.think, f.thinking ? 1 : 0, .1);
  tickTheme();
  if (t - lastT >= 32 && !document.hidden) { lastT = t; drawOrb(t); drawSpec(); drawScope(); }
  if (t - lastSlow > 250) {
    lastSlow = t;
    meter('mIn', 'vIn', sm.in); meter('mClean', 'vClean', sm.clean); meter('mOut', 'vOut', sm.out);
    ringV.util = lerp(ringV.util, ringT.util, .3); ringV.mem = lerp(ringV.mem, ringT.mem, .3);
    const ac = `rgb(${accentRGB.map(Math.round).join(',')})`;
    drawRing('rUtil', ringV.util, ringLbl.util, 'GPU LOAD', ac); drawRing('rMem', ringV.mem, ringLbl.mem, 'VRAM', '#a48bfa');
    spark('sE2E', series.e2e, 'rgba(52,211,153,.9)'); spark('sHdr', series.hdr, rgba(.9)); spark('sPause', series.pause, 'rgba(251,191,36,.9)');
    drawGpuHist();
    $('curSpk').textContent = S.typingUserSpk ? `speaker ${S.typingUserSpk}` : '—';
  }
  requestAnimationFrame(frame);
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
  el.childNodes[1].textContent = sc.enabled ? 'eyes' : 'eyes off';
  const meta = $('eyeMeta'), img = $('eyeImg');
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
  $('brainLbl').textContent = busy ? 'switching…' : remote ? 'remote llm' : 'local llm';
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
  try { const r = await fetch('/api/owner', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const st = await r.json(); S.snap = S.snap || {}; S.snap.owner = st; renderOwner(st); renderDock(); return st; } catch (_) { return null; }
}
function renderOwner(o) {
  if (!o) return;
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
  if (sl) {
    sl.addEventListener('input', () => { $('talkVal').textContent = Number(sl.value).toFixed(2); });
    sl.addEventListener('change', () => ownerPost({ talkativeness: { agent: S.persona || S.snap?.owner?.active, value: Number(sl.value) } }));
  }
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-mute]'); if (!b) return;
    const lab = b.getAttribute('data-mute');
    ownerPost(b.classList.contains('on') ? { unmute_speaker: lab } : { mute_speaker: lab });
  });
  fetch('/api/owner').then((r) => r.json()).then((o) => { S.snap = S.snap || {}; S.snap.owner = S.snap.owner || o; renderOwner(o); renderDock(); }).catch(() => {});
})();
(function wireEyes() {
  const el = $('pEyes'); if (!el) return;
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
  if (!id || id === 'you') return 'YOU';
  if (m.speakerName) _spkNames[id] = m.speakerName;
  if (_spkNames[id]) return _spkNames[id].toUpperCase();
  const n = /^S(\d+)$/.exec(id);
  return n ? 'GUEST ' + n[1] : id;
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
  const KEY = 'atlas.collapsed.v1';
  let saved; try { saved = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch { saved = null; }
  // first run: keep the busy diagnostics folded so the column reads cleanly
  const DEFAULT = ['GPU', 'Models', 'Threads', 'Model context', 'Audio routing'];
  const name = (p) => (p.querySelector('.ptitle')?.childNodes[0]?.textContent || '').trim();
  const closed = new Set(saved || DEFAULT);
  document.querySelectorAll('.side > .panel').forEach((p) => {
    if (closed.has(name(p))) p.classList.add('collapsed');
    p.querySelector('.ptitle').addEventListener('click', (e) => {
      if (e.target.closest('.tabs')) return;
      p.classList.toggle('collapsed');
      const now = [...document.querySelectorAll('.side > .panel.collapsed')].map(name);
      try { localStorage.setItem(KEY, JSON.stringify(now)); } catch {}
    });
  });
})();
