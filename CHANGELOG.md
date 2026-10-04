# Changelog

## 0.2.0 — memory, privacy and audio quality

**Memory**
- Each agent has its own long-term memory folder, created automatically when you make an agent
  (including with the + button). Agents never see each other's memories.
- Memory is a semantic hypergraph: lines from a call are embedded (bge-m3, on the CPU), linked,
  strengthened when they're used together and left to fade when they aren't. Relevant memories are
  brought back into a reply automatically.
- Privacy filter: lines containing phone numbers, emails, street addresses or API keys are never stored.
- Clear memory per agent: right-click an agent's icon, pick a window (last hour, 12 hours, day, week,
  or everything) and delete. Only that agent's memories are cleared.
- Deleting an agent archives its memories, so a new agent with the same name starts fresh.
- Saying someone's name in passing ("Sam, you're muted") no longer pulls up everything about Sam.
- Faster recall: memory search starts while people are still talking. In tests on real call audio,
  the typical added delay at the end of a turn fell from about 185 ms to under 1 ms.
- Agents can describe their own memory correctly when asked.

**Audio**
- Cleaner high voices: the 24 kHz to 48 kHz conversion now uses a proper filter (it left audible
  junk above 16 kHz), and loud syllables are smoothly limited instead of hard-clipped, which caused
  crackle.

**Fixes**
- Large memory batches could crash the embedder cache.
- An agent whose voice file was missing could stop the first boot.
- The installer no longer replaces an existing ATLAS shortcut that points at a different folder.

## 0.1.0 — first public build

- One-file Windows installer (`ATLAS-Setup-0.1.0.exe`) and `install.sh` for Ubuntu 22.04+.
- First-run setup wizard: language model (local via Ollama, or any OpenAI-compatible cloud provider
  with your own key), audio devices, voice import, agent creation.
- Local model setup: detects Ollama (or offers to install it), recommends a model sized for your GPU,
  downloads it with progress, and checks that it can follow ATLAS's reply format before saving.
- Cloud setup: the same reply-format check runs against your provider, so a key that works on a model
  that can't hold a conversation is caught during setup, not mid-call.
- Voice import from a short clip you have the right to use; a neutral starter voice is generated
  locally so ATLAS works before you import one.
- Turn-taking for group calls: agents answer when addressed and stay quiet when they aren't.
- Learns people's names from introductions instead of calling them "Speaker 3".
- Owner controls: instant mute, quiet mode, talkativeness, mute a specific speaker.
- Post-call report.
- The server only listens on this computer (127.0.0.1).
