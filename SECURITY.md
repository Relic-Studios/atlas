# Security

ATLAS runs entirely on your computer. It has no accounts, no telemetry and no bundled API keys.

## What ATLAS protects

| Risk | Protection |
|---|---|
| A web page you visit talks to ATLAS's local server | Host allowlist (blocks DNS rebinding), Origin check on every WebSocket and every state-changing request, no wildcard CORS |
| Other devices on your network reach ATLAS | Server binds to `127.0.0.1` only |
| Your cloud API key leaks | Encrypted at rest (Windows DPAPI; `0600` file on Linux); never sent back by the API; only sent to the provider URL you entered |
| Private details end up in long-term memory | Lines with phone numbers, emails, street addresses or keys are never stored; per-agent memory, wipe by time window |
| Malicious content in the desktop window | Electron sandbox, context isolation, no Node in the page, external links open in your browser |
| Known-vulnerable dependencies | Release builds run `pip-audit` and fail on any known CVE |
| Tampered downloads | Each release ships `SHA256SUMS.txt` |

## Verifying a download

```powershell
Get-FileHash .\ATLAS-Setup-<version>.exe -Algorithm SHA256
```
Compare with `SHA256SUMS.txt` on the release page. The installer is not yet code-signed, so
Windows SmartScreen may show "unrecognized app".

## Accepted findings

- **PYSEC-2026-3740 (nltk)**: file-sandbox bypass in `TransitionParser` model-path APIs. No fixed
  version exists. ATLAS never calls these APIs (nltk is only used for sentence splitting), so it is
  not reachable. Re-checked every release.

## Reporting a vulnerability

Please open a private report through GitHub: **Security → Report a vulnerability** on this
repository. Don't file public issues for security problems. We aim to reply within 7 days.
