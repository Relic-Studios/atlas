# Changelog

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
