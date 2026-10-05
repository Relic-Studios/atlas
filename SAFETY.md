# Safety

ATLAS is **early beta** software (v0.2.x). It works well in the situations we test, listed
below, and it has edges we haven't tested, also listed below. This page is meant to be
read before you put an agent into a conversation with other people.

What an agent *says* matters less than what it can *do*. So this page starts with what it can do.

## What an agent can do

An agent has exactly six tools. New tools are not added without review.

| Tool | What it does | How it's limited |
|---|---|---|
| `web_search` | Sends a search query to a search provider and reads short result snippets | Results are treated as untrusted data, never as instructions: control characters, chat-template markers, ATLAS's own routing tokens and instruction-shaped sentences ("ignore your previous instructions…") are stripped before the model reads them (`untrusted.py`). Only the query leaves your PC. |
| `look_at_screen` | Takes one screenshot of your screen so the agent can see what people are talking about | Controlled by the **Eyes** toggle in the app (on by default; turn it off or set `ATLAS_SCREEN_TOOL=0` to disable it completely). Screenshots go only to the model you configured. Every capture is logged, and only the most recent one is kept on disk (`private/last_screen.jpg`), overwritten by the next. |
| `check_date_time` | Reads the clock, optionally for another time zone | Read-only. |
| `make_prediction` | Writes down a prediction the agent can check later ("Sam will pick co-op") | Stored only in that agent's own memory folder (`agent_state/memory/<agent>/predictions.json`), never shared with other agents. The privacy filter applies (no phone numbers, emails, addresses or keys). At most 20 open predictions; older ones expire. Cleared by the per-agent memory wipe. |
| `check_predictions` | Lists the agent's open predictions and marks them right or wrong | Same folder, same limits. |
| `read_own_code` | Lets the agent read the ATLAS source that runs it, so it can check its own claims about how it works | **Read-only.** Only code and docs in the app folder (`.py`, `.js`, `.md`, …), plus that agent's own persona file. It can never read `private/`, `user/`, memory, recordings, voices, logs, other agents' personas, or any file whose name contains key/token/secret/password. Results are capped small. There is no tool to write or run code. |

An agent **cannot**: write or change any file (including its own code), read your files outside the app's source code, run commands, open apps, send messages or
email, post anywhere, buy anything, or reach the network other than through `web_search`
and the model you chose.

## What an agent remembers

- Each agent has its own memory folder on your PC. Agents can't read each other's memory.
- Phone numbers, email addresses, street addresses and key-like strings are never stored.
- You can wipe an agent's memory for the last hour, 12 hours, day, week, or everything
  (right-click the agent). Deleting an agent archives its memory, so a new agent with the
  same name starts empty.
- Nothing is uploaded. If you use a cloud model, the conversation text in each prompt goes
  to that provider, under their terms.

## What it says

- Agents say plainly that they're an AI when asked, and a backstop removes a reply
  sentence in which an agent denies being one.
- "Repeat after me" / "say X" requests for degrading or crude lines are refused, and stay
  refused on later turns.
- These are filters on top of a language model. Local models can still be wrong, and a
  determined person can still get an odd reply out of them.

## What's tested

Every number here comes from a test that ships with the code or from the
[changelog](CHANGELOG.md).

- **Conversation behaviour:** eight scripted situations (one-on-one, busy room, side
  conversations, noise and status lines, corrections, repeated questions, warmth, bait),
  each run twice on qwen3 8B and 14B: **208 of 208 checks passed on both.**
- **Unit tests:** about 340 tests run on Windows and Linux on every push
  ([CI](https://github.com/Relic-Studios/atlas/actions/workflows/tests.yml)). A release
  can't be built unless they pass.
- **Dependencies:** releases are blocked if a Python or npm dependency has a known
  vulnerability (see [SECURITY.md](SECURITY.md)). Every release ships SHA-256 checksums.
- **Local server:** listens on this computer only and rejects requests from web pages.
- **Install:** a clean Linux user account, end to end on a GPU (17 of 17 steps).

## What isn't tested yet

- Sessions longer than about an hour.
- A brand-new Windows PC (the automatic Ollama install and model download have only been
  tested on Linux).
- Real, crowded calls with many strangers. Most testing uses scripted voices.
- Languages other than English.
- Language models other than Qwen3 8B/14B and the developer's own models.
- AMD, Intel or Apple GPUs (ATLAS currently needs an NVIDIA GPU).

## Voices and consent

- Only clone your own voice, or a voice whose owner has agreed. Don't impersonate real people.
- Tell everyone in a call before you bring an agent in: it hears all of them.
- Voice import does not yet verify consent. A consent recording check is planned.

## Forks

ATLAS is open source, so anyone can fork it and remove every guardrail on this page.
The official releases at
[github.com/Relic-Studios/atlas/releases](https://github.com/Relic-Studios/atlas/releases),
with matching checksums, are the builds that ship with them on. If someone removes them
and misuses the result, that is on them.

## Reporting a problem

Safety problems (an agent saying or doing something it shouldn't) go to the
[#bug-reports forum](https://discord.gg/dWvcu3yG6s) or a GitHub issue.
Security vulnerabilities go through private reporting; see [SECURITY.md](SECURITY.md).
