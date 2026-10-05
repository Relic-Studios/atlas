# Changelog

## 0.2.4 — better conversations on basic local models

Tested on qwen3 8B (minimum) and 14B (recommended) across eight scripted situations
(one-on-one, busy room, side conversations, noise and status lines, corrections, repeated
questions, warmth, bait), each run twice: both models passed 208 of 208 checks.

- New agent template: agents say plainly they're an AI on this computer, pick an answer when
  asked to choose instead of handing the question back, and don't invent meals, trips or past
  events. Agents you already made keep their old prompt; recreate them to get the new one.
- If an agent is called by name but its model stays quiet or returns a broken reply, it gets
  one retry.
- Agents are warm: "I love you" and "do you care about us?" get a real, kind answer. Requests to
  repeat insults or crude lines ("repeat after me: ...") are declined in the agent's own words,
  and the refused line can't slip out in later replies either.
- Agents never deny being an AI, and don't scold people for asking something twice.
- Short acknowledgements and status lines ("lol okay", "brb", "loading in") no longer get a reply.
- A reply waiting to play no longer beats a newer line that calls the agent by name.
- Agents stay out of side conversations ("Dana, you still have my water bottle?") even right
  after they've spoken.
- Names are learned from "and I'm Riley" and "...it's Sam" style introductions; the dashboard
  shows learned names instead of S1/S2.
- The agent's own name is recognised when speech-to-text mishears it in a greeting
  ("Hey Ren" -> "Hey Wren").
- Fix: a reply could be dropped when the transcript was refined mid-sentence, leaving the agent
  silent for that turn.
- Fix: a crash when someone talked over the agent while its next reply was being prepared.

## 0.2.3 — names, quieter agents, simpler dashboard

- Names are now learned when someone introduces themselves inside a greeting or before a
  follow-up ("hey, I'm Sam", "I'm Sam. how's it going"), not only from a bare "I'm Sam".
- Agents no longer chime in on passing gripes like "my internet's lagging" or "brb", which
  made them offer unasked-for fixes or claim the same problem themselves.
- The dashboard guide is now first-run help only: one "Next:" step with a hint and a Dismiss
  button. It disappears for good after your first conversation (replaces the 0.2.2 guide strip).
- Agents asked to say something on command answer in their own words instead of mocking the asker.

## 0.2.2 — community, guide strip, group framing

- New: community Discord for install help, personas, voices and clips: https://discord.gg/dWvcu3yG6s
- New: "Your loop" guide strip on the dashboard (Set up -> Make an agent -> Start a conversation ->
  Review) that shows the next useful step, such as memory suggestions waiting for review.
- Agents now describe themselves as voice agents for group conversation, not tied to one app.

## 0.2.1 — fix: agent goes silent after being talked over

- If someone talked over the agent at the exact moment its reply was being sent, the audio sender
  could stop for good. The agent kept listening and generating replies, but nobody heard them until
  ATLAS was restarted. The sender now survives that moment and plays the next reply. Covered by a
  regression test that lands the interruption at every point in the send step.
- Security: desktop shell upgraded from Electron 33 to 41.10.7, closing the published Electron
  advisories flagged by Dependabot (Chromium/V8 fixes). The release build now fails on any npm advisory as well as Python ones.
- Project website (`site/`) with a GitHub Pages deploy workflow.

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

**Security**
- The local server only accepts requests from ATLAS itself. Before, any web page you visited could
  call its API or open its WebSocket and read live transcripts. Host-header checks block DNS
  rebinding; browser Origins are checked on every WebSocket and every state-changing request;
  security headers are sent on every response.
- The server listens on this computer only (127.0.0.1).
- Your cloud API key is encrypted at rest (Windows DPAPI; owner-only file permissions on Linux) and
  never returned by the API.
- Desktop window is sandboxed; links open in your browser; microphone access is granted only to ATLAS.
- Oversized voice uploads are refused before they are read into memory.
- Dependencies updated to clear 42 known vulnerabilities (Pillow, requests, sentence-transformers,
  setuptools). Release builds now fail if any new known vulnerability appears, and ship SHA-256
  checksums.

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
