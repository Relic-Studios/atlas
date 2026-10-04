/* ATLAS desktop shell — spawns the Python backend, waits for it, then loads the UI. */
const { app, BrowserWindow, session, globalShortcut, shell } = require('electron');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = process.env.ATLAS_PORT || 8000;
const APP_URL = `http://localhost:${PORT}/static/atlas.html`;
const HEALTH_URL = `http://localhost:${PORT}/`;
const SERVER_DIR = path.resolve(__dirname, '..'); // dir containing server.py
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

function startBackend() {
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

async function restartBackend() {
  if (!mainWindow) return;
  await mainWindow.loadFile(path.join(__dirname, 'splash.html'));
  if (!(await healthCheck())) startBackend();
  if (await waitForBackend()) await mainWindow.loadURL(APP_URL);
  else showBackendDown('Backend failed to start');
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

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1000,
    minHeight: 640,
    backgroundColor: '#0b0d10',
    autoHideMenuBar: true,
    title: 'ATLAS',
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true, webSecurity: true },
    show: false,
  });
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

app.on('before-quit', () => {
  quitting = true;
  globalShortcut.unregisterAll();
  // Tree-kill: backend.kill() on Windows only kills python.exe and orphans the
  // RealtimeSTT worker (which keeps Whisper resident on the GPU).
  if (backend) {
    spawnSync('taskkill', ['/PID', String(backend.pid), '/T', '/F'], { windowsHide: true });
    backend = null;
  }
});
