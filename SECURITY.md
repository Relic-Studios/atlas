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

- **PyTorch 2.8.0** (installed by `install.ps1` from PyTorch's CUDA index, so outside the `pip-audit` gate):
  GHSA-63cw-57p8-fm3p (code execution when `torch.load` opens an untrusted checkpoint) and memory-safety
  or denial-of-service bugs in specific functions given crafted input (GHSA-qfhq-4f3w-5fph, GHSA-rrmf-rvhw-rf47,
  GHSA-vgrw-7cvw-pwgx, PYSEC-2025-203/204/206, PYSEC-2026-139, PYSEC-2026-2286). ATLAS loads only the models
  on its fixed download list, never a checkpoint someone hands it (imported voices are audio clips), and doesn't
  pass outside input to those functions. An update to a fixed PyTorch is planned once it passes the speech and
  voice tests.

## Reporting a vulnerability

Please open a private report through GitHub: **Security → Report a vulnerability** on this
repository. Don't file public issues for security problems. We aim to reply within 7 days.
