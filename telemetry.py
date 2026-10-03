"""Live system telemetry for the ATLAS dashboard.

Read-only by design: nothing here sits on the realtime audio/LLM path.
  * TelemetryLogHandler turns the pipeline's existing log lines into structured
    events (decisions, TTFT, turn ends, aborts, gates, errors) and keeps a
    bounded tail of raw log lines.
  * Telemetry.snapshot() samples GPU / Ollama / process / pipeline state.
  * Telemetry.fast() samples audio meters + a log-band spectrum (cheap, ~15 Hz).
Served by server.py at GET /api/telemetry and WS /telemetry (separate from
/ws so a dashboard connection never aborts a live generation).
"""
from __future__ import annotations

import collections
import json
import logging
import re
import threading
import time
import urllib.request

import numpy as np

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SKIP = re.compile(r"TTS ALLOWED|State +ToClient|Potential sentence end|Queuing text for pause|"
                   r"Starting pause calculation|telemetry|GET /api/telemetry|capture\(\) takes")

# (event kind, regex) — first match wins.
_PATTERNS = [
    ("decision", re.compile(r"Model decision=(?P<action>\w+) target=(?P<target>\S*) header_ms=(?P<ms>[\d.]+)")),
    ("ttft", re.compile(r"\[Gen (?P<gen>\d+)\] LLM Worker: TTFT: (?P<s>[\d.]+)s")),
    ("gen_start", re.compile(r"\[Gen (?P<gen>\d+)\] Preparing new generation for: '(?P<text>.*?)(?:\.\.\.)?'$")),
    ("abort", re.compile(r"Gen (?P<gen>\d+) Text \('(?P<text>.*?)\.\.\.'\) different enough")),
    ("speaker", re.compile(r"Speaker for turn: (?P<speaker>\S+)")),
    ("user_final", re.compile(r"FINAL USER REQUEST \(STT Callback\): (?P<text>.*)")),
    ("agent_final", re.compile(r"FINAL ASSISTANT ANSWER \(Sending\): (?P<text>.*)")),
    ("turn_end", re.compile(r"USER TURN END|Recording stopped")),
    ("tts_start", re.compile(r"\[Gen (?P<gen>\d+)\] Quick TTS Worker: Synthesizing")),
    ("pause", re.compile(r"Calculated pauses: .*Final=(?P<final>[\d.]+) for \"(?P<text>.*)\" \(Prob=(?P<prob>[\d.]+)\)")),
    ("gate", re.compile(r"(?P<text>HOLD \(quiet mode\).*|.*explicit silence request.*|.*veto.*|.*held fragment.*|"
                        r".*filler-only.*|.*Model HOLD retired.*|.*barge.?in.*)", re.I)),
    ("persona", re.compile(r"Switched persona to '(?P<name>\w+)'")),
    ("voice", re.compile(r"(?:Voice override set to|self-reference ->) '?(?P<name>[\w-]+)'?")),
]


class Telemetry:
    def __init__(self, max_events: int = 400, max_lines: int = 300):
        self.app = None
        self.started = time.time()
        self.events: collections.deque = collections.deque(maxlen=max_events)
        self.lines: collections.deque = collections.deque(maxlen=max_lines)
        self.seq = 0
        self.lock = threading.Lock()
        self.counters = collections.Counter()
        self.header_ms: collections.deque = collections.deque(maxlen=120)
        self.ttft_s: collections.deque = collections.deque(maxlen=120)
        self.e2e_s: collections.deque = collections.deque(maxlen=120)
        self.pause_s: collections.deque = collections.deque(maxlen=120)
        self._turn_end_at = None
        self._last_speaker = None
        self._nvml = None
        self._ollama_cache = (0.0, {})
        self._proc = None
        self._gpu_hist: collections.deque = collections.deque(maxlen=120)

    # ------------------------------------------------------------------ events
    def _push(self, kind: str, **data) -> None:
        with self.lock:
            self.seq += 1
            self.events.append({"seq": self.seq, "t": time.time(), "kind": kind, **data})
            self.counters[kind] += 1

    def ingest(self, record: logging.LogRecord) -> None:
        try:
            msg = _ANSI.sub("", record.getMessage()).strip()
        except Exception:  # noqa: BLE001
            return
        if not msg or _SKIP.search(msg):
            return
        now = time.time()
        with self.lock:
            self.seq += 1
            self.lines.append({"seq": self.seq, "t": now, "lvl": record.levelname,
                               "src": record.name[:12], "msg": msg[:400]})
        if record.levelno >= logging.ERROR:
            self._push("error", text=msg[:300], src=record.name)
            return
        for kind, rx in _PATTERNS:
            m = rx.search(msg)
            if not m:
                continue
            d = {k: v for k, v in m.groupdict().items() if v is not None}
            if kind == "decision":
                ms = float(d["ms"]); self.header_ms.append(ms)
                self.counters["decision_" + d["action"]] += 1
                self._push(kind, action=d["action"], target=d["target"], ms=ms)
            elif kind == "ttft":
                s = float(d["s"])
                if s < 60:
                    self.ttft_s.append(s)
                self._push(kind, gen=int(d["gen"]), s=s)
            elif kind == "turn_end":
                self._turn_end_at = now
            elif kind == "tts_start":
                if self._turn_end_at is not None and now - self._turn_end_at < 30:
                    self.e2e_s.append(now - self._turn_end_at)
                    self._push("latency", s=now - self._turn_end_at)
                self._turn_end_at = None
                self._push(kind, gen=int(d["gen"]))
            elif kind == "pause":
                self.pause_s.append(float(d["final"]))
                self.counters["pause"] += 1   # too chatty for the event feed
            elif kind == "speaker":
                self._last_speaker = d["speaker"]
                self._push(kind, speaker=d["speaker"])
            elif kind == "user_final":
                self._push(kind, text=d["text"][:300], speaker=self._last_speaker)
            else:
                self._push(kind, **{k: (v[:300] if isinstance(v, str) else v) for k, v in d.items()})
            return

    def events_since(self, seq: int) -> list:
        with self.lock:
            return [e for e in self.events if e["seq"] > seq]

    def lines_since(self, seq: int) -> list:
        with self.lock:
            return [l for l in self.lines if l["seq"] > seq]

    # ------------------------------------------------------------------ samplers
    def _gpu(self) -> dict:
        try:
            if self._nvml is None:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import pynvml
                pynvml.nvmlInit()
                self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            nv, h = self._nvml
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            name = nv.nvmlDeviceGetName(h)
            g = {
                "name": name.decode() if isinstance(name, bytes) else name,
                "mem_used": mem.used / 2**30, "mem_total": mem.total / 2**30,
                "util": util.gpu, "mem_util": util.memory,
                "temp": nv.nvmlDeviceGetTemperature(h, 0),
                "power": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "power_limit": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
                "clock": nv.nvmlDeviceGetClockInfo(h, 0),
            }
            try:
                procs = nv.nvmlDeviceGetComputeRunningProcesses(h) + nv.nvmlDeviceGetGraphicsRunningProcesses(h)
                g["procs"] = self._name_gpu_procs(procs)
            except Exception:  # noqa: BLE001
                g["procs"] = []
            self._gpu_hist.append((time.time(), g["util"], g["mem_used"]))
            return g
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:120]}

    @staticmethod
    def _name_gpu_procs(procs) -> list:
        import psutil
        out = {}
        for p in procs:
            try:
                name = psutil.Process(p.pid).name()
            except Exception:  # noqa: BLE001
                name = str(p.pid)
            used = (p.usedGpuMemory or 0) / 2**30
            out[name] = out.get(name, 0.0) + used
        return sorted(({"name": k, "gb": v} for k, v in out.items() if v > 0.01), key=lambda x: -x["gb"])[:8]

    def _ollama(self) -> dict:
        ts, cached = self._ollama_cache
        if time.time() - ts < 3.0:
            return cached
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=1.5) as r:
                data = json.loads(r.read().decode())
            models = [{"name": m.get("name"), "vram_gb": (m.get("size_vram") or 0) / 2**30,
                       "ctx": m.get("context_length"), "expires": m.get("expires_at"),
                       "quant": (m.get("details") or {}).get("quantization_level"),
                       "params": (m.get("details") or {}).get("parameter_size")}
                      for m in data.get("models", [])]
            res = {"up": True, "models": models}
        except Exception as e:  # noqa: BLE001
            res = {"up": False, "error": str(e)[:120], "models": []}
        self._ollama_cache = (time.time(), res)
        return res

    def _process(self) -> dict:
        try:
            import psutil
            if self._proc is None:
                self._proc = psutil.Process()
                self._proc.cpu_percent(None)
            p = self._proc
            kids = p.children(recursive=True)
            rss = p.memory_info().rss + sum(k.memory_info().rss for k in kids if k.is_running())
            return {"cpu": p.cpu_percent(None), "sys_cpu": psutil.cpu_percent(None),
                    "rss_gb": rss / 2**30, "threads": p.num_threads(), "children": len(kids),
                    "sys_mem": psutil.virtual_memory().percent}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:120]}

    def _pipeline(self) -> dict:
        st = getattr(self.app, "state", None)
        spm = getattr(st, "SpeechPipelineManager", None)
        if spm is None:
            return {}
        gen = spm.running_generation
        out = {
            "persona": spm.current_persona,
            "voice": getattr(spm.audio, "current_voice", None),
            "model": getattr(spm.llm, "model", None),
            "gen_counter": spm.generation_counter,
            "gen": None if gen is None else {
                "id": gen.id, "text": (gen.text or "")[:160],
                "decision": getattr(getattr(gen, "decision", None), "action", None),
                "target": getattr(getattr(gen, "decision", None), "target", None),
                "quick": (gen.quick_answer or "")[:200], "aborting": gen.abortion_started},
            "llm_active": spm.llm_generation_active,
            "tts_active": spm.tts_quick_generation_active or spm.tts_final_generation_active,
            "history_len": len(spm.history),
            "latency_floor_ms": getattr(spm, "full_output_pipeline_latency", None),
        }
        convo = getattr(spm, "convo", None)
        if convo is not None:
            try:
                out["convo"] = convo.stats()
                out["room_note"] = convo.state_note()[:600]
                book = getattr(spm, "people", None)
                out["names"] = {k: book.display(k) for k in list(convo.roster)} if book else {}
                out["roster"] = [{"id": k, "name": out["names"].get(k, ""),
                                  "turns": v.turns, "words": v.words,
                                  "idle_s": round(convo.clock() - v.last_seen, 1),
                                  "addressed": v.addressed_agent, "reactions": v.reactions,
                                  "last": v.last_line[:120]}
                                 for k, v in list(convo.roster.items())[:12]]
                out["ledger"] = [{"role": e.role, "who": e.speaker or "?", "text": e.text[-220:]}
                                 for e in list(convo.ledger)[-16:]]
            except Exception as e:  # noqa: BLE001
                out["convo_error"] = str(e)[:120]
        fl = getattr(spm, "floor", None)
        if fl is not None:
            try:
                out["floor"] = {"partner": fl.active_partner(), "quiet": fl.is_quiet(),
                                "quiet_left": max(0.0, fl.quiet_until - fl.clock()),
                                "room_size": fl.room_size()}
            except Exception as e:  # noqa: BLE001
                out["floor"] = {"error": str(e)[:120]}
        if getattr(spm, "boards", None) is not None:
            try:
                out["tasks"] = spm.board().snapshot()
            except Exception as e:  # noqa: BLE001
                out["tasks"] = [{"error": str(e)[:120]}]
        th = getattr(spm, "threads", None)
        if th is not None:
            try:
                now = th.clock()
                out["threads"] = {
                    "active": None if th.active is None else {"speaker": th.active.speaker, "text": th.active.text[:140]},
                    "open": sorted(({"speaker": t.speaker, "text": t.text[:140], "score": round(t.score(now), 2),
                                     "reasons": list(t.reasons)} for t in th.open.values()),
                                   key=lambda x: -x["score"])[:8]}
            except Exception as e:  # noqa: BLE001
                out["threads"] = {"error": str(e)[:120]}
        return out

    def _speakers(self) -> list:
        st = getattr(self.app, "state", None)
        br = getattr(st, "CallBridge", None)
        dz = getattr(br, "diarizer", None) or getattr(st, "Diarizer", None)
        if dz is None:
            return []
        now = time.time()
        try:
            return [{"label": p.label, "speech_s": round(p.speech_seconds, 1), "sentences": p.sentence_count,
                     "idle_s": round(now - p.last_seen_at, 1)} for p in list(dz._profiles)]
        except Exception:  # noqa: BLE001
            return []

    def _stt(self) -> dict:
        st = getattr(self.app, "state", None)
        aip = getattr(st, "AudioInputProcessor", None)
        tr = getattr(aip, "transcriber", None)
        if tr is None:
            return {}
        rec = False
        try:
            rec = bool(tr._get_recorder_param("is_recording", False)) if hasattr(tr, "_get_recorder_param") else False
        except Exception:  # noqa: BLE001
            pass
        return {"recording": rec, "silence": bool(getattr(tr, "silence_active", False)),
                "partial": (getattr(tr, "realtime_text", None) or "")[-200:]}

    @staticmethod
    def _stats(values) -> dict:
        v = sorted(values)
        if not v:
            return {"n": 0}
        return {"n": len(v), "last": list(values)[-1], "p50": v[len(v) // 2],
                "p90": v[min(len(v) - 1, int(len(v) * 0.9))], "min": v[0], "max": v[-1],
                "series": list(values)[-60:]}

    @staticmethod
    def _screen():
        try:
            import screen as _scr
            s = _scr.status()
            return {"enabled": s.get("enabled"), "count": s.get("count"), "last_ts": s.get("last_ts"),
                    "last_reason": s.get("last_reason"), "last_size": s.get("last_size")}
        except Exception:  # noqa: BLE001
            return None

    def _llm_backend(self):
        try:
            llm = self.app.state.SpeechPipelineManager.llm
            rs = getattr(llm, "remote", None)
            return {"mode": "remote" if rs is not None else "local",
                    "switching": bool(getattr(self.app.state, "llm_switching", False)),
                    "remote": rs.status() if rs is not None else None}
        except Exception:  # noqa: BLE001
            return None

    def _owner(self):
        try:
            import owner_controls as _owner
            return _owner.snapshot()
        except Exception:  # noqa: BLE001
            return None

    def snapshot(self) -> dict:
        st = getattr(self.app, "state", None)
        br = getattr(st, "CallBridge", None)
        ca = getattr(br, "call_audio", None)
        return {
            "t": time.time(), "uptime": time.time() - self.started,
            "gpu": self._gpu(), "gpu_hist": [[round(t, 1), u, round(m, 2)] for t, u, m in list(self._gpu_hist)[-90:]],
            "ollama": self._ollama(), "proc": self._process(),
            "pipeline": self._pipeline(), "speakers": self._speakers(), "stt": self._stt(),
            "bridge": None if br is None else {"in": getattr(br, "input_device", None),
                                               "out": getattr(br, "output_device", None),
                                               "aec": bool(getattr(ca, "_aec", None)),
                                               "playout": (ca.playout_stats() if hasattr(ca, "playout_stats") else None)},
            "screen": self._screen(),
            "llm_backend": self._llm_backend(),
            "owner": self._owner(),
            "counters": dict(self.counters),
            "latency": {"header_ms": self._stats(self.header_ms), "ttft_s": self._stats(self.ttft_s),
                        "e2e_s": self._stats(self.e2e_s), "pause_s": self._stats(self.pause_s)},
        }

    def fast(self) -> dict:
        st = getattr(self.app, "state", None)
        br = getattr(st, "CallBridge", None)
        ca = getattr(br, "call_audio", None)
        spec = []
        if ca is not None:
            blk = getattr(ca, "last_clean_block", None)
            if blk is not None and len(blk) >= 256:
                x = np.asarray(blk, dtype=np.float32) * np.hanning(len(blk)).astype(np.float32)
                mag = np.abs(np.fft.rfft(x))
                n = len(mag)
                edges = np.unique(np.geomspace(2, n - 1, 33).astype(int))
                bands = [float(mag[a:b].mean()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
                spec = [round(float(np.clip((20 * np.log10(b + 1e-6) + 20) / 50, 0, 1)), 3) for b in bands]
        stt = self._stt()
        speaking = False
        try:
            speaking = bool(br.is_speaking()) if br is not None else False
        except Exception:  # noqa: BLE001
            pass
        spm = getattr(st, "SpeechPipelineManager", None)
        return {"type": "fast", "t": time.time(),
                "in": getattr(ca, "level_in", 0.0), "clean": getattr(ca, "level_clean", 0.0),
                "out": getattr(ca, "level_out", 0.0), "spec": spec,
                "speaking": speaking, "recording": stt.get("recording", False),
                "partial": stt.get("partial", ""),
                "thinking": bool(spm and spm.llm_generation_active)}


class TelemetryLogHandler(logging.Handler):
    def __init__(self, telemetry: Telemetry):
        super().__init__(level=logging.INFO)
        self.telemetry = telemetry

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.telemetry.ingest(record)
        except Exception:  # noqa: BLE001 - never let telemetry break logging
            pass


TELEMETRY = Telemetry()


def install(app) -> Telemetry:
    TELEMETRY.app = app
    root = logging.getLogger()
    if not any(isinstance(h, TelemetryLogHandler) for h in root.handlers):
        root.addHandler(TelemetryLogHandler(TELEMETRY))
    return TELEMETRY
