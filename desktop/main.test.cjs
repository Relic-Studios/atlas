const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const { EventEmitter } = require('node:events');
const source = fs.readFileSync(path.join(__dirname, 'main.js'), 'utf8');

function harness(ownsInstance = true, ready = false) {
  const stats = { spawns: 0, quits: 0, loads: 0, kills: 0, requests: [], callEndReply: '{"ended": true, "start_pass": false}' };
  const app = new EventEmitter();
  app.requestSingleInstanceLock = () => ownsInstance;
  app.setAppUserModelId = (id) => { stats.appId = id; };
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
  }, request(opts, callback) {
    stats.requests.push(opts.path);
    const req = new EventEmitter();
    req.destroy = () => {};
    req.end = () => queueMicrotask(() => {
      if (stats.callEndReply === null) { req.emit('error', new Error('down')); return; }
      const res = new EventEmitter();
      callback(res);
      res.emit('data', stats.callEndReply);
      res.emit('end');
    });
    return req;
  }};
  const context = vm.createContext({
    require(name) {
      if (name === 'electron') return { app, BrowserWindow: class {}, session: {},
        globalShortcut: { unregisterAll() {} }, shell: {} };
      if (name === 'child_process') return { spawn(cmd, args) { stats.spawns++;
        if (args && args[0] === 'memory_writer.py') { stats.passes = (stats.passes || 0) + 1;
          stats.passArgs = args.slice(1); }
        const child = new EventEmitter(); child.stdout = new EventEmitter();
        child.stderr = new EventEmitter(); return child; }, spawnSync() { stats.kills++; } };
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

const DISPLAYS = [{ workArea: { x: 0, y: 0, width: 2560, height: 1400 } },
  { workArea: { x: 2560, y: 0, width: 3440, height: 1400 } }];

test('saved window bounds are reused only on a connected display and at a usable size', () => {
  const { context } = harness();
  const pick = (b) => context.pickBounds({ bounds: b }, DISPLAYS);
  // (the object comes from the vm realm, so compare its JSON)
  assert.equal(JSON.stringify(pick({ x: 2600, y: 40, width: 1600, height: 1000 })), '{"x":2600,"y":40,"width":1600,"height":1000}');
  assert.equal(pick({ x: 7000, y: 40, width: 1600, height: 1000 }), null);   // that monitor is gone
  assert.equal(pick({ x: 100, y: -200, width: 1600, height: 1000 }), null);  // title bar above the screen
  assert.equal(pick({ x: 100, y: 40, width: 800, height: 1000 }), null);     // smaller than the minimum
  assert.equal(pick({ x: 100, y: 40, width: 'wide', height: 1000 }), null);
  assert.equal(context.pickBounds({}, DISPLAYS), null);
  assert.equal(context.pickBounds(null, DISPLAYS), null);
});

test('zoom steps by 10% between 80% and 150%, and 0 resets it', () => {
  const { context } = harness();
  assert.equal(context.stepZoom(1, 1), 1.1);
  assert.equal(context.stepZoom(1.2, 1), 1.3);
  assert.equal(context.stepZoom(1.5, 1), 1.5);
  assert.equal(context.stepZoom(0.8, -1), 0.8);
  assert.equal(context.stepZoom(1.3, 0), 1);
  assert.equal(context.stepZoom('junk', 1), 1.1);
  assert.equal(context.clampZoom(3), 1.5);
  assert.equal(context.clampZoom(undefined), 1);
});

test('quitting ends the call through the API before stopping the backend', async () => {
  const { context, stats } = harness();
  vm.runInContext('backend = { pid: 4242 }', context);
  let prevented = 0;
  await context.beforeQuit({ preventDefault() { prevented++; } });
  assert.equal(prevented, 1);
  assert.deepEqual(stats.requests, ['/api/call/end']);
  assert.equal(stats.kills, 1);
  assert.equal(stats.quits, 1);
  await context.beforeQuit({ preventDefault() { prevented++; } });   // the second quit goes through
  assert.equal(prevented, 1);
  assert.equal(stats.kills, 1);
});

test('a backend that does not answer is still stopped', async () => {
  const { context, stats } = harness();
  stats.callEndReply = null;
  vm.runInContext('backend = { pid: 4242 }', context);
  await context.beforeQuit({ preventDefault() {} });
  assert.equal(stats.kills, 1);
  assert.equal(stats.quits, 1);
});

test('a backend this app did not start is left running', async () => {
  const { context, stats } = harness();
  await context.beforeQuit({ preventDefault() {} });
  assert.deepEqual(stats.requests, []);
  assert.equal(stats.kills, 0);
});

test('the post-call pass starts after the backend stops, only when the server says so', async () => {
  const { context, stats } = harness();
  stats.callEndReply = '{"ended": true, "start_pass": true}';
  vm.runInContext('backend = { pid: 4242 }', context);
  await context.beforeQuit({ preventDefault() {} });
  assert.equal(stats.kills, 1);
  assert.equal(stats.passes, 1);
  const second = harness();
  vm.runInContext('backend = { pid: 4242 }', second.context);
  await second.context.beforeQuit({ preventDefault() {} });      // start_pass false
  assert.equal(second.stats.passes, undefined);
});

test('the pass at quit talks to the model the server named', async () => {
  const { context, stats } = harness();
  stats.callEndReply = '{"ended": true, "start_pass": true, "pass_args": ["--mode", "remote", "--url", "http://box:8000", "--model", "m"]}';
  vm.runInContext('backend = { pid: 4242 }', context);
  await context.beforeQuit({ preventDefault() {} });
  assert.deepEqual(Array.from(stats.passArgs), ['--mode', 'remote', '--url', 'http://box:8000', '--model', 'm']);
});
