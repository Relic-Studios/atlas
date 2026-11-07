<p align="center"><img src="docs/images/banner.png" alt="ATLAS: local voice agents for group conversation" width="100%"></p>

<p align="center">
  <a href="https://github.com/Relic-Studios/atlas/releases/tag/v0.7.0-beta.2"><b>Download 0.7 beta for Windows</b></a> ·
  <a href="#linux">Linux</a> ·
  <a href="#requirements">Requirements</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="SAFETY.md">Safety</a> ·
  <a href="https://discord.gg/dWvcu3yG6s">Community Discord</a>
</p>

<p align="center"><a href="https://github.com/Relic-Studios/atlas/actions/workflows/tests.yml"><img src="https://github.com/Relic-Studios/atlas/actions/workflows/tests.yml/badge.svg" alt="tests"></a></p>

# ATLAS

Local voice agents for group conversation. They listen to everyone in the room or call,
answer when they're talked to, and speak in a voice you choose.
Listening and the voice run on your PC. Choose a local language model or connect your own provider.
Local mode needs no account, and ATLAS bundles no API keys.

> **Early beta (0.7.0).** Core installed pipelines have been verified; broader hardware and external integrations still need coverage.
> Read [SAFETY.md](SAFETY.md) for exactly what an agent can do, what's tested, and what isn't.

[![Watch the live local 14B demo](docs/images/atlas-live-local14b.png)](docs/media/atlas-live-local14b.mp4)

An 83-second continuous run of the installed app using **local Qwen3 14B** and Juniper Take A.
Human inputs use one synthetic voice; every assistant reply is generated live. The run includes a contextual
follow-up, staying quiet for an aside, a real clock lookup, and an audible interruption followed by a return to dinner.
Only idle time after the final reply was trimmed. [Capture and audio provenance](docs/media/atlas-live-local14b-provenance.json).

<p align="center">
  <img src="docs/images/voices.jpg" width="100%" alt="Four voices over a real one-minute conversation, the agent in teal">
</p>

- **Runs on your PC.** Listening, the voice and (if you choose) the language model all run locally.
- **Your voice, your agent.** Import a 10–20 second clip, describe a personality, and ATLAS drafts the agent for you to edit.
- **Knows when to talk.** In a group conversation it answers when it's addressed and stays quiet when it isn't.
  Short in banter, fuller when you ask it something real.
- **Learns names.** It picks up people's names from introductions instead of calling them "Speaker 3".
- **Knows your voice.** Press **This is me** once and every agent recognizes you in later calls.
- **Remembers the calls, not the people.** Each agent keeps short memories of the calls it was in, and facts only about you.
- **You stay in control.** Instant mute, quiet mode, a talkativeness slider, mute a specific speaker, and a report after each conversation.

## Install

### Windows

1. Download **`ATLAS-Setup-<version>.exe`** (about 140 MB) from the
   [0.7 beta release](https://github.com/Relic-Studios/atlas/releases/tag/v0.7.0-beta.2) and run it.
2. Setup checks your graphics card first, so a PC that can't run ATLAS finds out before any download.
3. Click through the pages (license, folder, shortcuts) and **Install**. Setup shows each step, how much is downloaded
   and the time left: about 7 GB in all (PyTorch, ATLAS's packages, and the speech and voice models), 20 to 30 minutes
   on a 100 Mbit/s connection. You can keep using your PC.
4. At the end, leave **Launch ATLAS** ticked. When the speech and voice downloads succeed, those models are ready for
   the first launch. Setup warns if a download needs to finish later. Choose your language model and audio devices;
   choosing an optional local language model downloads that model separately.

No admin rights are needed, and nothing is installed system-wide: ATLAS goes to your own Program Files
(`%LOCALAPPDATA%\Programs\ATLAS`) with its own copy of Python and of the desktop app, so you don't need Python or
Node.js. With the default folder, Setup needs about 17 GB free on that drive (the app, its Python environment, the
models and download caches); the Ready page checks every drive it uses.

- **Stop** stops the install at any point. Run Setup again to finish; steps that are done are skipped.
- If a step fails (for example the connection drops), Setup says what went wrong and offers **Try again** and
  **Open log**.
- **Updating:** run the new `ATLAS-Setup` over the old install. It closes ATLAS if it's open, keeps your Python
  environment, models and data, and installs only what changed.

> Windows may show a SmartScreen warning because the installer isn't code-signed yet.
> Click **More info → Run anyway**.

From source instead: clone this repo and double-click **Install ATLAS.cmd**
(or `powershell -ExecutionPolicy Bypass -File install.ps1`). It installs Python 3.12 and Node.js with winget if
they're missing. Options: `-SkipModels`, `-NoShortcut`. Each run adds to `install.log`.

**Uninstall:** Windows Settings > Apps > **ATLAS** > Uninstall, or **Uninstall ATLAS** in the Start menu.
It removes the app, its Python, its Python environment, the desktop app and the shortcuts, and asks whether to keep
your data (settings, voices, agents, memories, call recordings) for a later reinstall. The downloaded speech and voice
models stay in the shared Hugging Face cache, since other programs may use them.

### Linux

Ubuntu 22.04+ with an NVIDIA driver installed:

```bash
git clone https://github.com/Relic-Studios/atlas.git && cd atlas
bash install.sh   # asks for sudo once for system packages
```

To uninstall: `bash uninstall.sh` in that folder (asks before deleting your data).

## First launch

<p align="center"><img src="docs/images/setup-wizard.png" width="70%" alt="ATLAS first-time setup"></p>

The setup wizard walks you through:

1. **Language model**
   - **Local (recommended with 12 GB+ of VRAM):** ATLAS finds Ollama (or offers to install it),
     recommends a model sized for your GPU, downloads it with progress, and checks it can follow
     ATLAS's reply format before saving it.
   - **Cloud:** any OpenAI-compatible provider (OpenRouter, Together, Groq, OpenAI, your own vLLM server).
     Paste your own key; it's stored only in `user/settings.json` on this PC and never shown again in full.
     The same reply-format check runs against your provider.
   - **Subscription account:** Claude Code or Codex through the installed official CLI. Connect opens its official
     sign-in flow; Verify checks an existing session. No token copying or API billing fallback. This beta's inference
     protocol checks support Claude Code 2.1.281 and Codex 0.146.1; other versions show an update message.
     A verified but exhausted account stays selected and displays its limit until you explicitly retry.
   - **Vision:** the local selector also offers vision-capable models. Text-only or unverified models clearly show
     that Eyes is unavailable. Plugin and model capability requirements still apply.
2. **Audio:** pick the input the agent listens on and the output it speaks to.
   To bring it into an online voice call (any app), route the call into a virtual cable (e.g. VB-Audio Voicemeeter)
   and pick that cable here.
3. **Voice and agent:** Juniper and her bundled voice are ready by default. Start with her, or optionally
   import a clean voice reference and create an agent. Drafting uses the model you have loaded; you can edit it before saving.

All optional plugins start uninstalled and off. In **Plugins → Marketplace**, install a bundled plugin,
configure it, then enable it in Installed. Installing does not enable it. Keys, compatible models or extra
software are required for some plugins; the bundled plugin setup guide explains each requirement.

You can change all of this later from the dashboard.

## Requirements

- Windows 10/11 or Ubuntu 22.04+, 16 GB RAM, about 17 GB free disk to install (plus the local model, if you use one).
- **An NVIDIA GPU is required:** GeForce RTX 20-series or newer. GTX 10-series (Pascal) and older cards aren't supported. Listening and the voice use about 6 GB of VRAM.
- **NVIDIA driver 570.65 or newer** (the one for CUDA 12.8). If yours is older, Setup says so; updates are free at
  [nvidia.com/drivers](https://www.nvidia.com/drivers).

| Your GPU | What works |
|---|---|
| 8–11 GB (e.g. RTX 3060 Ti, 4060) | ATLAS with a **cloud** model |
| 12–15 GB (e.g. RTX 3060 12 GB, 4070) | Fully local with Qwen3 8B (minimum tested) |
| 16 GB+ (e.g. RTX 4080, 4090, 5090) | Fully local with Qwen3 14B (recommended) |

Smaller local models (4B and under) were benchmarked and can't keep up with a group call.

## Community

Questions, install help, persona and voice tips, clips: join the [ATLAS community Discord](https://discord.gg/dWvcu3yG6s).
Bugs can also go in [GitHub issues](https://github.com/Relic-Studios/atlas/issues); security reports go through [SECURITY.md](SECURITY.md), not public channels.

## Memory

Each agent remembers the calls it was in: what you talked about, what was funny, how it went.
Memories are made after a call ends, by a low-priority pass on your PC, so they never slow a reply down.
A memory comes up in conversation only when it clearly fits, and at most once every 10 minutes unless you ask.

- **Experiences, not people.** Names of the people in the call are taken out before anything is stored.
  Agents keep no facts about anyone but you, and no one's voice, from one call to the next.
- **Facts about you** come only from things you said yourself (that's what **This is me** is for).
- **"Remember that…"** saves something on the spot, **"forget that"** removes it, and **"off the record"**
  keeps a stretch of the call out of memory until you say you're back.
- Phone numbers, emails, addresses and keys are never stored.
- Each agent keeps its own memories; only the facts about you are shared between agents.
- With a cloud model, memories are only made if you tick **Allow cloud processing**, and only the
  transcript with names taken out is sent.
- Until you turn on **Use memories**, agents only use what you asked them to remember.
- Right-click an agent to open its **Memory** window: see what it keeps, delete a single memory,
  or clear the last hour, 12 hours, day, week, or everything.

The full rules are in [SAFETY.md](SAFETY.md#what-an-agent-remembers).

## Tests

```
pip install -r requirements-ci.txt
python -m unittest discover -s tests
```

About 900 unit tests cover turn-taking, name learning, bait and repeat guards, memory isolation and privacy, setup, and the local-only security layer. They need no GPU or models and run on every push ([tests workflow](https://github.com/Relic-Studios/atlas/actions/workflows/tests.yml)). Releases are only built when they pass.

## License

ATLAS is licensed under the **GNU Affero General Public License v3.0 or later**
(AGPL-3.0-or-later). See [LICENSE](LICENSE). Copyright (C) 2026 Relic Studios.

In plain terms (the LICENSE file is what counts):
- You may use, study, modify and share ATLAS, including commercially.
- If you distribute a modified version, **or run one as a service other people use
  over a network**, you must release your source code under the same license.
- There is no warranty.

The name "ATLAS" as used for this project and the Relic Studios logo are not
covered by the code license. Forks must use a different name and make clear they
are not the official release. See [SAFETY.md](SAFETY.md) for the guardrails the
official builds ship with and the use rules for the official community.

## Privacy

Audio, transcripts and memories stay on your PC. The server only listens on this computer (127.0.0.1).
Web searches (when an agent looks something up) and cloud model calls (only if you chose a cloud provider)
are the only things that leave it.

Only use voices you have the right to use.
