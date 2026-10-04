// Record the real ATLAS UI during a demo call: offscreen Electron, 30 fps paint
// frames saved as JPEG with wall-clock timestamps (ms) in the file name.
//   electron tools/demo/capture.js <url> <frames_dir> <done_file> [width] [height]
// Stops when <done_file> exists (written by demo_call.py) or after 15 minutes.
const { app, BrowserWindow } = require('electron');
const fs = require('fs');
const path = require('path');

const [url, dir, doneFile, W = '1600', H = '900'] = process.argv.slice(2);
fs.mkdirSync(dir, { recursive: true });
app.disableHardwareAcceleration();

app.whenReady().then(() => {
  const w = new BrowserWindow({
    width: +W, height: +H, show: false, backgroundColor: '#0b0f17',
    webPreferences: { offscreen: true, contextIsolation: true },
  });
  w.webContents.setFrameRate(30);
  let n = 0, last = null;
  w.webContents.on('paint', (_e, _dirty, image) => { last = image; });
  // Write at a steady 30 fps from the latest painted frame (paint only fires on change).
  const timer = setInterval(() => {
    if (!last) return;
    fs.writeFileSync(path.join(dir, `${Date.now()}.jpg`), last.toJPEG(88));
    n++;
    if (fs.existsSync(doneFile)) { clearInterval(timer); console.log('frames', n); setTimeout(() => app.quit(), 500); }
  }, 1000 / 30);
  setTimeout(() => { clearInterval(timer); console.log('timeout, frames', n); app.quit(); }, 15 * 60 * 1000);
  w.loadURL(url);
});
