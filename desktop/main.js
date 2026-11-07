/* ATLAS desktop shell — spawns the Python backend, waits for it, then loads the UI. */
const { app, BrowserWindow, session, globalShortcut, shell, screen } = require('electron');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = process.env.ATLAS_PORT || 8000;
const APP_URL = `http://localhost:${PORT}/static/atlas.html`;
const HEALTH_URL = `http://localhost:${PORT}/`;
const SERVER_DIR = path.resolve(__dirname, '..'); // dir containing server.py
const APP_ID = 'RelicStudios.ATLAS';
const APP_ICON = path.join(SERVER_DIR, 'static', 'favicon.ico');
if (process.platform === 'win32') app.setAppUserModelId(APP_ID);
// Explicit interpreter: in git-bash `python3` is the WindowsApps shim. Prefer the
// per-user 3.12 install when present, else whatever `python` is on PATH.
const fs = require('fs');
const USER_PY = path.join(process.env.LOCALAPPDATA || '', 'Programs/Python/Python312/python.exe');
// Installed builds carry their own venv (install.ps1); dev uses the user 3.12.
const VENV_PY = process.platform === 'win32'
  ? path.join(SERVER_DIR, '.venv', 'Scripts', 'python.exe')
  : path.join(SERVER_DIR, '.venv', 'bin', 'python');
const PYTHON = process.env.ATLAS_PYTHON || (fs.existsSync(VENV_PY) ? VENV_PY
  : (fs.existsSync(USER_PY) ? USER_PY : (process.platform === 'win32' ? 'python' : 'python3')));

let backend = null;
let mainWindow = null;
let quitting = false;
let restartPromise = null;
const ownsInstance = app.requestSingleInstanceLock();
if (!ownsInstance) app.quit();
app.on('second-instance', () => {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
});

function startBackend() {
  // A starting backend is already owned even before its HTTP port is ready.
  if (backend) return;
  // Archive the previous log, then tee backend output to server.log so crashes
  // are diagnosable even when ATLAS was launched from the desktop shortcut.
  const fs = require('fs'), logPath = path.join(SERVER_DIR, 'server.log');
  try {
    if (fs.existsSync(logPath)) {
      fs.mkdirSync(path.join(SERVER_DIR, 'logs'), { recursive: true });
      fs.copyFileSync(logPath, path.join(SERVER_DIR, 'logs', `server_${Date.now()}.log`));
    }
    rotateLogs(path.join(SERVER_DIR, 'logs'));
  } catch (e) { console.error('log archive failed', e); }
  const logStream = fs.createWriteStream(logPath, { flags: 'w' });
  backend = spawn(PYTHON, ['-u', 'server.py'], { cwd: SERVER_DIR, windowsHide: true,
    env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' } });
  backend.stdout.on('data', (d) => logStream.write(d));
  backend.stderr.on('data', (d) => logStream.write(d));
  backend.on('exit', (code, signal) => {
    console.log(`[backend] exited (code=${code}, signal=${signal})`);
    backend = null;
    if (!quitting && code === 75) { restartBackend(); return; }  // settings applied, relaunch
    if (!quitting) showBackendDown(`Backend stopped (exit code ${code}${signal ? ', ' + signal : ''})`);
  });
}

// Keep the newest 6 archived logs plain (replay/grep), gzip older ones in the
// background. Nothing is deleted: archived calls are replay material.
function rotateLogs(dir) {
  const fs = require('fs'), zlib = require('zlib');
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.log'))
    .map((f) => ({ f, t: fs.statSync(path.join(dir, f)).mtimeMs })).sort((a, b) => b.t - a.t);
  for (const { f } of files.slice(6)) {
    const src = path.join(dir, f), dst = src + '.gz';
    if (fs.existsSync(dst)) continue;
    fs.createReadStream(src).pipe(zlib.createGzip()).pipe(fs.createWriteStream(dst))
      .on('finish', () => { try { fs.unlinkSync(src); } catch (_) {} })
      .on('error', (e) => console.error('gzip failed', f, e));
  }
}

// Last error-ish lines of server.log, for the "backend down" screen.
function lastErrors() {
  try {
    const txt = require('fs').readFileSync(path.join(SERVER_DIR, 'server.log'), 'utf8')
      .replace(/\x1b\[[0-9;]*m/g, '').replace(/\u0000/g, '');
    const lines = txt.split(/\r?\n/).filter(Boolean);
    const tb = lines.map((l, i) => (/Traceback|\bERRO\b|Error:|Exception|FATAL|CUDA error|out of memory/.test(l) ? i : -1)).filter((i) => i >= 0);
    const from = tb.length ? Math.max(0, tb[tb.length - 1] - 4) : Math.max(0, lines.length - 25);
    return lines.slice(from, from + 30).join('\n');
  } catch (e) { return 'server.log unreadable: ' + e.message; }
}

async function showBackendDown(msg) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  await mainWindow.loadFile(path.join(__dirname, 'splash.html'));
  const js = `document.getElementById('status').textContent=${JSON.stringify(msg)};` +
    `document.querySelector('.dot').style.background='#f87171';` +
    `const e=document.getElementById('err');e.textContent=${JSON.stringify(lastErrors())};e.style.display='block';` +
    `document.getElementById('restart').style.display='inline-block';`;
  mainWindow.webContents.executeJavaScript(js).catch(() => {});
}

function restartBackend() {
  if (restartPromise) return restartPromise;
  restartPromise = (async () => {
    if (!mainWindow || quitting) return;
    await mainWindow.loadFile(path.join(__dirname, 'splash.html'));
    if (!(await healthCheck())) startBackend();
    if (await waitForBackend()) await mainWindow.loadURL(APP_URL);
    else showBackendDown('Backend failed to start');
  })().finally(() => { restartPromise = null; });
  return restartPromise;
}

function healthCheck() {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, (res) => { res.resume(); resolve(res.statusCode === 200); });
    req.on('error', () => resolve(false));
    req.setTimeout(1500, () => { req.destroy(); resolve(false); });
  });
}

async function waitForBackend(timeoutMs = 10 * 60 * 1000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (await healthCheck()) return true;
    await new Promise((r) => setTimeout(r, 1500));
  }
  return false;
}


// ---- window memory and zoom --------------------------------------------------------------------------------------
// The window reopens where the owner left it, at the zoom the owner chose. Zoom is Chromium's page zoom (it scales
// everything evenly), applied to every page of the backend (dashboard, Plugins, Setup) each time one loads; the local
// splash page stays at 100%. Nothing here runs at load time: the state is read when the window is created.
const ZOOM_MIN = 0.8, ZOOM_MAX = 1.5, ZOOM_STEP = 0.1, MIN_W = 1000, MIN_H = 640;

// Saved bounds are reused only if they are big enough and the point just below the middle of the title bar lies in a
// connected display's work area; otherwise the window opens at the default size.
function pickBounds(saved, displays) {
  const b = saved && saved.bounds;
  if (!b || ![b.x, b.y, b.width, b.height].every(Number.isFinite) || b.width < MIN_W || b.height < MIN_H) return null;
  const px = b.x + b.width / 2, py = b.y + 15;
  const on = (displays || []).some((d) => { const w = d && d.workArea;
    return w && px >= w.x && px < w.x + w.width && py >= w.y && py < w.y + w.height; });
  return on ? { x: b.x, y: b.y, width: b.width, height: b.height } : null;
}

// A zoom factor rounded to a 10% step and kept between 80% and 150% (anything unreadable is 100%).
function clampZoom(z) {
  const n = Math.round((Number(z) || 1) * 10) / 10;
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, n));
}
// One zoom step in a direction: 1 in, -1 out, 0 back to 100%.
function stepZoom(z, dir) { return dir === 0 ? 1 : clampZoom(clampZoom(z) + dir * ZOOM_STEP); }

function statePath() { return path.join(app.getPath('userData'), 'window-state.json'); }
function readWindowState() {
  try { const s = JSON.parse(require('fs').readFileSync(statePath(), 'utf8')); return s && typeof s === 'object' ? s : {}; }
  catch (_) { return {}; }
}
let windowState = { zoom: 1 }, stateTimer = null;
function saveWindowState(win) {
  if (!win || win.isDestroyed()) return;
  windowState = { bounds: win.getNormalBounds(), maximized: win.isMaximized(), zoom: clampZoom(windowState.zoom) };
  try { require('fs').writeFileSync(statePath(), JSON.stringify(windowState)); } catch (e) { console.error('window state not saved', e); }
}
function saveWindowStateSoon(win) { clearTimeout(stateTimer); stateTimer = setTimeout(() => saveWindowState(win), 500); }

function onBackendPage(win) {
  try { return new URL(win.webContents.getURL()).origin === new URL(APP_URL).origin; } catch (_) { return false; }
}
function setZoom(win, z) {
  windowState.zoom = z;
  if (onBackendPage(win)) win.webContents.setZoomFactor(z);
  saveWindowStateSoon(win);
}
function wireWindowState(win) {
  win.on('resize', () => saveWindowStateSoon(win));
  win.on('move', () => saveWindowStateSoon(win));
  win.on('close', () => { clearTimeout(stateTimer); saveWindowState(win); });
  const wc = win.webContents;
  wc.on('did-finish-load', () => wc.setZoomFactor(onBackendPage(win) ? (windowState.zoom || 1) : 1));
  // Ctrl (or Cmd) with = + numpad-add: in; - numpad-subtract: out; 0 numpad-0: 100%. preventDefault() also keeps the
  // default menu's zoom from applying on top; its reload and DevTools shortcuts still work.
  wc.on('before-input-event', (event, input) => {
    if (input.type !== 'keyDown' || !(input.control || input.meta) || input.alt) return;
    const k = input.key, c = input.code;
    const dir = (k === '=' || k === '+' || c === 'NumpadAdd') ? 1
      : (k === '-' || k === '_' || c === 'NumpadSubtract') ? -1 : (k === '0' || c === 'Numpad0') ? 0 : null;
    if (dir === null) return;
    event.preventDefault();
    if (onBackendPage(win)) setZoom(win, stepZoom(windowState.zoom, dir));
  });
  // Ctrl + mouse wheel: one step per 100 ms at most (a trackpad sends many events)
  let lastWheel = 0;
  wc.on('zoom-changed', (_e, direction) => {
    const now = Date.now();
    if (now - lastWheel < 100 || !onBackendPage(win)) return;
    lastWheel = now;
    setZoom(win, stepZoom(windowState.zoom, direction === 'in' ? 1 : -1));
  });
}

async function createWindow() {
  const saved = readWindowState();
  windowState = { zoom: clampZoom(saved.zoom) };
  const bounds = pickBounds(saved, screen ? screen.getAllDisplays() : []) || { width: 1440, height: 900 };
  mainWindow = new BrowserWindow({
    ...bounds,
    minWidth: 1000,
    minHeight: 640,
    backgroundColor: '#0b0d10',
    autoHideMenuBar: true,
    title: 'ATLAS',
    icon: APP_ICON,
    titleBarStyle: 'hidden',
    titleBarOverlay: { color: '#0b0d10', symbolColor: '#b7c2d0', height: 32 },
    webPreferences: { preload: path.join(__dirname, 'shell-preload.js'), contextIsolation: true, nodeIntegration: false, sandbox: true, webSecurity: true },
    show: false,
  });
  if (process.platform === 'win32') mainWindow.setAppDetails({
    appId: APP_ID,
    appIconPath: APP_ICON,
    appIconIndex: 0,
    relaunchCommand: `"${process.execPath}" "${__dirname}"`,
    relaunchDisplayName: 'ATLAS',
  });
  wireWindowState(mainWindow);
  if (saved.maximized && pickBounds(saved, screen ? screen.getAllDisplays() : [])) mainWindow.maximize();
  await mainWindow.loadFile(path.join(__dirname, 'splash.html'));
  mainWindow.once('ready-to-show', () => mainWindow.show());
  // Security: the window may only show ATLAS's own pages. Anything else (a link in
  // a transcript, a provider's key page) opens in the user's browser instead.
  const appOrigin = new URL(APP_URL).origin;
  const isAppUrl = (u) => { try { const x = new URL(u); return x.origin === appOrigin || x.protocol === 'file:'; } catch (_) { return false; } };
  const openOutside = (u) => { try { const x = new URL(u); if (x.protocol === 'https:' || x.protocol === 'http:') shell.openExternal(x.href); } catch (_) {} };
  mainWindow.webContents.setWindowOpenHandler(({ url }) => { openOutside(url); return { action: 'deny' }; });
  mainWindow.webContents.on('will-navigate', (ev, url) => {
    if (url.startsWith('atlas://restart')) { ev.preventDefault(); restartBackend(); return; }
    if (!isAppUrl(url)) { ev.preventDefault(); openOutside(url); }
  });

  const ok = await waitForBackend();
  if (ok) {
    await mainWindow.loadURL(APP_URL);
  } else {
    showBackendDown('Backend failed to start');
  }
}

app.whenReady().then(async () => {
  if (!ownsInstance) return;
  // Grant microphone/media permission (getUserMedia in the renderer).
  // Microphone only, and only for ATLAS's own pages.
  const appOrigin = new URL(APP_URL).origin;
  session.defaultSession.setPermissionRequestHandler((wc, permission, cb) => {
    let own = false;
    try { own = new URL(wc.getURL()).origin === appOrigin; } catch (_) {}
    cb(own && (permission === 'media' || permission === 'mediaKeySystem'));
  });

  // Reuse a backend that is already running; otherwise start our own.
  if (!(await healthCheck())) startBackend();
  createWindow();

  // Owner panic button: instant hard mute toggle from any window.
  globalShortcut.register('CommandOrControl+Shift+M', () => {
    const body = JSON.stringify({ mute: 'toggle' });
    const req = http.request({ host: '127.0.0.1', port: PORT, path: '/api/owner', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (res) => res.resume());
    req.on('error', () => {});
    req.end(body);
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Quit: end the call through the API first, so the server writes the call's end and clears what the call held
// (docs/designs/experience_memory.md, D2), then stop the backend. Only a backend this app started is stopped.
function endCall(timeoutMs = 3000) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v) => { if (!done) { done = true; resolve(v); } };
    const req = http.request({ host: '127.0.0.1', port: PORT, path: '/api/call/end', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': 2 } }, (res) => {
      let body = '';
      if (res.setEncoding) res.setEncoding('utf8');
      res.on('data', (c) => { body += c; });
      res.on('end', () => { try { finish(JSON.parse(body)); } catch (_) { finish(null); } });
    });
    req.on('error', () => finish(null));
    setTimeout(() => { try { req.destroy(); } catch (_) {} finish(null); }, timeoutMs);
    req.end('{}');
  });
}

function startMemoryPass(args) {
  // args: what /api/call/end said the pass talks to (the live model's mode, URL and name; never a key)
  const passArgs = Array.isArray(args) && args.length ? args.map(String) : ['--mode', 'remote'];
  try {
    const child = spawn(PYTHON, ['memory_writer.py', ...passArgs],
      { cwd: SERVER_DIR, windowsHide: true, detached: true, stdio: 'ignore' });
    if (child && child.unref) child.unref();
  } catch (e) { console.error('memory pass not started', e); }
}

function stopBackend() {
  // Tree-kill: backend.kill() on Windows only kills python.exe and orphans the
  // RealtimeSTT worker (which keeps Whisper resident on the GPU).
  if (backend) {
    spawnSync('taskkill', ['/PID', String(backend.pid), '/T', '/F'], { windowsHide: true });
    backend = null;
  }
}

let quitStage = 0;   // 0: not started, 1: ending the call, 2: done
function beforeQuit(event) {
  quitting = true;
  globalShortcut.unregisterAll();
  if (backend && quitStage === 0) {
    if (event && event.preventDefault) event.preventDefault();
    quitStage = 1;
    return endCall(3000).then((res) => {
      stopBackend();
      // The post-call pass, after the server is gone (docs/designs/experience_memory.md, D3): only when the server
      // said so (never with a model on this PC), detached so it finishes after the window has closed.
      if (res && res.start_pass) startMemoryPass(res.pass_args);
      quitStage = 2;
      app.quit();
    });
  }
  if (quitStage === 1) {               // still ending the call: hold the quit until it is done
    if (event && event.preventDefault) event.preventDefault();
    return Promise.resolve();
  }
  stopBackend();
  return Promise.resolve();
}

app.on('before-quit', (event) => { beforeQuit(event); });
