const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const { EventEmitter } = require('node:events');
const source = fs.readFileSync(path.join(__dirname, 'main.js'), 'utf8');

function harness(ownsInstance = true, ready = false) {
  const stats = { spawns: 0, quits: 0, loads: 0 };
  const app = new EventEmitter();
  app.requestSingleInstanceLock = () => ownsInstance;
  app.quit = () => { stats.quits++; };
  app.whenReady = () => ready ? Promise.resolve() : new Promise(() => {});
  const fakeFs = { existsSync: () => false, readdirSync: () => [],
    createWriteStream: () => ({ write() {} }) };
  const fakeHttp = { get(url, callback) {
    const req = new EventEmitter();
    req.setTimeout = () => {};
    req.destroy = () => {};
    queueMicrotask(() => {
      if (stats.spawns) callback({ statusCode: 200, resume() {} });
      else req.emit('error', new Error('not ready'));
    });
    return req;
  }};
  const context = vm.createContext({
    require(name) {
      if (name === 'electron') return { app, BrowserWindow: class {}, session: {},
        globalShortcut: { unregisterAll() {} }, shell: {} };
      if (name === 'child_process') return { spawn() { stats.spawns++;
        const child = new EventEmitter(); child.stdout = new EventEmitter();
        child.stderr = new EventEmitter(); return child; }, spawnSync() {} };
      if (name === 'fs') return fakeFs;
      if (name === 'http') return fakeHttp;
      return require(name);
    }, process, __dirname, console, setTimeout, Buffer,
    testWindow: { loadFile: async () => {}, loadURL: async () => { stats.loads++; } },
  });
  vm.runInContext(source, context);
  return { context, stats };
}

test('a second desktop launch quits without starting a backend', async () => {
  const { stats } = harness(false, true);
  await new Promise(setImmediate);
  assert.equal(stats.quits, 1);
  assert.equal(stats.spawns, 0);
});

test('a backend still booting cannot be spawned twice', () => {
  const { context, stats } = harness();
  context.startBackend();
  context.startBackend();
  assert.equal(stats.spawns, 1);
});

test('concurrent restart requests share one startup and one UI reload', async () => {
  const { context, stats } = harness();
  vm.runInContext('mainWindow = testWindow', context);
  const first = context.restartBackend();
  const second = context.restartBackend();
  assert.equal(first, second);
  await first;
  assert.equal(stats.spawns, 1);
  assert.equal(stats.loads, 1);
});
