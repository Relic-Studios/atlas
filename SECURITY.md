# Security

ATLAS runs entirely on your computer. It has no accounts, no telemetry and no bundled API keys.

## What ATLAS protects

| Risk | Protection |
|---|---|
| A web page you visit talks to ATLAS's local server | Host allowlist (blocks DNS rebinding), Origin check on every WebSocket and every state-changing request, no wildcard CORS |
| Other devices on your network reach ATLAS | Server binds to `127.0.0.1` only |
| Your cloud API key leaks | Encrypted at rest (Windows DPAPI; `0600` file on Linux); never sent back by the API; only sent to the provider URL you entered |
| Private details end up in long-term memory | Memories are made after a call, on your PC (a cloud provider only if you allow it); names of people in the call are removed before anything is stored; lines with phone numbers, emails, street addresses or keys are never stored; "off the record" keeps a stretch out; view and delete memories in each agent's Memory window, by time window or all at once |
| Malicious content in the desktop window | Electron sandbox, context isolation, no Node in the page, external links open in your browser |
| Known-vulnerable dependencies | Release builds run `pip-audit` on the Python requirements and `npm audit` on the desktop shell, and fail on any known CVE not listed under Accepted findings |
| Tampered downloads | Each release ships `SHA256SUMS.txt` |

## Verifying a download

```powershell
Get-FileHash .\ATLAS-Setup-<version>.exe -Algorithm SHA256
```
Compare with `SHA256SUMS.txt` on the release page. The installer is not yet code-signed, so
Windows SmartScreen may show "unrecognized app".

## Accepted findings

- **PyTorch 2.10.0** (installed by `install.ps1` / `install.sh` from PyTorch's CUDA 12.8 index, so outside the
  `pip-audit` gate): two low-severity, local-only advisories remain. GHSA-rrmf-rvhw-rf47 is memory corruption
  when `torch.jit.script` compiles crafted code; it is fixed in PyTorch 2.13, which has no CUDA 12.8 build.
  PYSEC-2026-139 is unsafe deserialization when loading a crafted `.pt2` archive; no fix has been released.
  ATLAS never passes outside code to `torch.jit.script` (libraries script only their own built-in functions),
  never loads `.pt2` archives, and loads only the models on its fixed download list (imported voices are audio
  clips). The other advisories against the earlier 2.8.0, including GHSA-63cw-57p8-fm3p (`torch.load` of an
  untrusted checkpoint), are fixed in 2.10.0.

## Reporting a vulnerability

Please open a private report through GitHub: **Security → Report a vulnerability** on this
repository. Don't file public issues for security problems. We aim to reply within 7 days.
