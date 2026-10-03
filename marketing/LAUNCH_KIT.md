# ATLAS launch kit

Every claim here is backed by a test that was actually run. Numbers come from
`tests/bench_min_model.py`, `tests/bench_quality.py` and `tests/e2e_new_user.py` (dev repo).
If you add a claim, add the test first.

## One-liners

- **Tagline:** Voice agents that join your calls and run on your own PC.
- **Short (store/listing, ~140 chars):** ATLAS puts a voice agent in your group calls. It listens,
  knows when it's being talked to, and answers in a voice you choose. Fully local.
- **Elevator:** ATLAS is a desktop app that adds a talking AI to your voice calls. You give it a voice
  from a short clip and a personality in plain words. It hears everyone, learns their names, answers
  when someone talks to it and stays out of the way when they don't. The speech recognition, the
  voice and (if you want) the language model all run on your graphics card. No account, no
  subscription, no keys hidden in the app.

## What it does (measured)

| Claim | Evidence |
|---|---|
| Stays quiet when it isn't addressed | Qwen3 14B: 0 butt-ins across the turn-taking benchmark; Qwen3 8B: 0 |
| Answers when addressed | 14B: 0 missed; 8B: 3 of 12 missed |
| Actually does what it's asked | 14B: 12/12 requests; 8B: 10/12 |
| Replies are its own words, not copied prompt lines | 14B: 29/30 clean replies |
| Fast decisions | 8B: 0.43 s median decision on an RTX 4090 |
| Installs from nothing | Clean Ubuntu 24.04 user, GPU: 17/17 setup steps pass, including real model download |
| No hidden keys | Fresh install starts in setup mode with no provider configured; the only key is the one the user pastes |

## Feature bullets

- **Bring a voice.** Import a short clip and its transcript; ATLAS speaks in that voice.
- **Describe a personality.** Write a few lines, let ATLAS draft the rest, edit, save.
- **Built for groups.** Knows who's talking, learns names, doesn't talk over people.
- **Your call, your rules.** Instant mute, quiet mode, talkativeness slider, mute a person.
- **Local first.** Ollama with a model picked for your GPU, or any OpenAI-compatible cloud with your key.
- **After-call report.** Who it talked to, when it held back, what it said.

## Requirements (say this plainly everywhere)

- NVIDIA GPU required. 12 GB for fully local (8B), 16 GB+ recommended (14B), 8 GB with a cloud model.
- Windows 10/11 or Ubuntu 22.04+. No macOS. No CPU-only mode (the voice engine is CUDA-only).

## Do not claim

- "Runs on any PC" / "no GPU needed" — false.
- "Human-level conversation" — no test supports it.
- "Works with every model" — models under 8B were benchmarked and fail.
- Anything about specific characters or celebrity voices. Users must only use voices they have rights to.
- "Signed / trusted installer" — until a code-signing certificate is bought, SmartScreen will warn.

## Assets still to make (after a clean public install, never from the dev build)

- Screenshots: setup wizard (each step), dashboard mid-call, owner controls, post-call report.
- 60–90 s demo video: install → import voice → create agent → agent answering in a real call.
- Social card 1200×630.
