# Copilot Prompt API

[![CI](https://github.com/nikkel/copilot-sdk-fastapi-wrapper/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nikkel/copilot-sdk-fastapi-wrapper/actions/workflows/ci.yml)
[![Python 3.11 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Audited with pip-audit](https://img.shields.io/badge/audited%20with-pip--audit-4B8BBE)](https://github.com/pypa/pip-audit)
[![Last commit](https://img.shields.io/github/last-commit/nikkel/copilot-sdk-fastapi-wrapper)](https://github.com/nikkel/copilot-sdk-fastapi-wrapper/commits/main)

A FastAPI wrapper around the [github-copilot-sdk](https://pypi.org/project/github-copilot-sdk/) that exposes GitHub Copilot as a local HTTP API. Send prompts, stream responses, and list available models — all from any HTTP client.

---

## Models and cost

Since **1 June 2026** Copilot uses usage-based billing: there are **no free models any more**. Every chat
request uses AI Credits from your plan's monthly allowance (1 credit = $0.01; e.g. Copilot Pro includes
1,500 credits/month). Prices are per million tokens, see
[Models and pricing for GitHub Copilot](https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing).

Cheapest models available to this account (checked 2026-10-15 via `GET /models`):

| Model ID | Input / output per 1M tokens | Notes |
|---|---|---|
| `gpt-6-luna` | $0.10 / $0.50 | **default**, cheapest available |
| `mai-code-1.1-flash` | $0.20 / $1.20 | |
| `gpt-5.6-luna` | $0.20 / $1.20 | |
| `gpt-5-mini` | $0.25 / $2.00 | the old "free" model, now billed |

A short prompt costs a tiny fraction of a credit with `gpt-6-luna`. `GET /models` lists what your plan offers;
model availability and prices change, so check the pricing page above.

`/chat`, `/chat/stream` and `/agent/step` only accept models your account lists (checked against a list refreshed
every 5 minutes), because Copilot would otherwise silently answer an unknown model with a different, possibly
pricier one. `auto` works if your account lists it.

| Status | Meaning |
|---|---|
| `422` | Unknown model. `detail.valid_models` has up to 10 valid IDs, closest match first; no session is created |
| `503` | The model list couldn't be fetched, so the request wasn't sent |

---

## Prerequisites

### 1 — Install the GitHub CLI

```bash
brew install gh
```

### 2 — Authenticate with your GitHub account

```bash
gh auth login
```

Follow the interactive prompts. Choose **GitHub.com** → **HTTPS** → authenticate via browser.

### 3 — Install Python dependencies

Requires **Python 3.11+**.

`github-copilot-sdk` 1.x downloads the Copilot CLI runtime it needs on first start (cached in
`~/Library/Caches/github-copilot-sdk/` on macOS), so the old `gh-copilot` extension is no longer required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock
```

Use a virtual environment: an older `github-copilot-sdk` (0.x) installed globally makes `main.py` fail with
`ImportError: cannot import name 'PermissionDecisionReject'`.

`requirements.lock` holds the exact version and hashes of every package, so an install can't differ from what was
tested. It's generated from the direct dependencies in `requirements.txt`:
- `github-copilot-sdk` 1.0.22, pinned because model validation uses its experimental `rpc.models` API — Python SDK that runs and talks to the Copilot CLI runtime
- `fastapi` (≥ 0.143.2) — web framework
- `uvicorn[standard]` (≥ 0.54.0) — ASGI server
- `pydantic` (≥ 2.14.1) — request/response models

The client runs in the SDK's **empty mode** (data in `~/.copilot-fastapi-wrapper`, override with
`COPILOT_WRAPPER_HOME`): no workspace/environment context or custom instructions are added to prompts.

Sessions are **answer-only**: Copilot's built-in tools (shell, file edits, …) are disabled and every
permission request is refused, so prompts containing untrusted text can't make it act on your machine.

---

## Running the Server

The server won't start without an API key. Generate one and start the server in the same shell:

```bash
export WRAPPER_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn main:app --reload
```

The server starts on **http://localhost:8000**. To keep the same key between shells, save it in a secrets manager or
a private file that isn't committed, and export it from there. The key is what stops anything else that can reach the
port from spending your credits, so treat it like a password. To replace it, set a new value and restart the server.

Send the key with every request:

```bash
curl -s http://localhost:8000/models -H "Authorization: Bearer $WRAPPER_API_KEY"
```

| Setting | Required | Description |
|---|---|---|
| `WRAPPER_API_KEY` | ✅ | The key callers must send as `Authorization: Bearer <key>` |
| `WRAPPER_ALLOWED_HOSTS` | ❌ | Comma-separated hostnames the server answers to, without ports. Replaces the default `localhost,127.0.0.1`, so list those too if you still need them. Wildcards are refused at startup |

Every request is checked in this order:

| Status | Meaning |
|---|---|
| `400` | The `Host` header isn't in `WRAPPER_ALLOWED_HOSTS`. This applies to every path, including `/health`. It stops web pages reaching the server through DNS rebinding |
| `401` | The key is missing or wrong. This applies to every path except `/health` |

| URL | Description |
|---|---|
| http://localhost:8000/docs | **Swagger UI** — interactive API explorer (needs the key) |
| http://localhost:8000/redoc | ReDoc — alternative docs viewer (needs the key) |
| http://localhost:8000/openapi.json | Raw OpenAPI schema (needs the key) |

---

## Swagger UI (Interactive Docs)

The docs pages need the key too, and a browser can't send it when you open a page directly, so
**http://localhost:8000/docs** returns `401` in a normal browser tab. To use them, add an
`Authorization: Bearer <key>` header with a browser extension that sets request headers, limited to this server so the
key isn't sent to other sites.

To read the schema without a browser:

```bash
curl -s http://localhost:8000/openapi.json -H "Authorization: Bearer $WRAPPER_API_KEY" | python3 -m json.tool
```

---

## API Reference

### `POST /chat` — Send a prompt (blocking)

Returns the full response once the model has finished generating.

**Request body:**

```json
{
  "prompt": "string",
  "model": "gpt-6-luna",
  "reasoning_effort": null
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | ✅ | The message to send to Copilot |
| `model` | string | ❌ | Model ID (default: `gpt-6-luna`) |
| `reasoning_effort` | string \| null | ❌ | `"low"` / `"medium"` / `"high"` / `"xhigh"` — only for models that support reasoning |
| `system` | string \| null | ❌ | Your own system prompt; **replaces** Copilot's built-in one (fewer tokens, follows your instructions) |

**Response:**

```json
{
  "response": "The assistant's full reply",
  "model": "gpt-6-luna"
}
```

---

### `POST /chat/stream` — Send a prompt (streaming)

Returns the response as a stream of [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events). Each event contains one chunk of text. The final event is `data: [DONE]`.

Same request body as `/chat`.

---

### `POST /agent/step` — One step of an agent (native tool calling)

For apps that implement their own tools (e.g. a browser agent). The tools are registered with Copilot as real
functions, so the model uses native tool calling; **nothing is executed here**: each call is recorded, the turn
ends, and the calls are returned for your app to run. Send the results back in the next step's `prompt`.

```json
{
  "system": "You operate a web browser…",
  "prompt": "User: find the phone number of …",
  "tools": [{"type": "function", "function": {"name": "search", "description": "…", "parameters": {…}}}],
  "model": "gpt-6-luna"
}
```

Response: `{"content": "…", "tool_calls": [{"name": "search", "arguments": {"query": "…"}}], "model": "gpt-6-luna"}`.
An empty `tool_calls` means the model answered in `content`.

Tool names must be 1-64 letters, digits, underscores or hyphens; anything else returns `422` and no session is created.

---

### `GET /models` — List available models

Returns the models advertised by the connected Copilot CLI.

**Response:**

```json
{
  "models": [{"id": "gpt-6-luna", ...}, {"id": "gpt-5-mini", ...}, "..."]
}
```

---

### `GET /health` — Health check

The only route that needs no API key. The `Host` check still applies.

```json
{ "status": "ok" }
```

---

## Examples

All examples read the key from `WRAPPER_API_KEY`, so set it in the shell that runs them.

### curl

#### Basic prompt with the default model (`gpt-6-luna`)

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $WRAPPER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a Python generator?"}'
```

#### Choosing another model (`gpt-5-mini`)

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $WRAPPER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function that flattens a nested list",
    "model": "gpt-5-mini"
  }'
```

#### Streaming response with `gpt-6-luna`

```bash
curl -s -N -X POST http://localhost:8000/chat/stream \
  -H "Authorization: Bearer $WRAPPER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain async/await in Python step by step"}'
```

Each line of output looks like:

```
data: Async/await is
data:  a way to write
data:  concurrent code...
data: [DONE]
```

#### List available models

```bash
curl -s http://localhost:8000/models -H "Authorization: Bearer $WRAPPER_API_KEY" | python3 -m json.tool
```

---

### Python (`requests`)

#### Blocking request

```python
import os
import requests

response = requests.post(
    "http://localhost:8000/chat",
    headers={"Authorization": f"Bearer {os.environ['WRAPPER_API_KEY']}"},
    json={
        "prompt": "Give me a one-liner to reverse a string in Python",
        "model": "gpt-6-luna",
    },
)
print(response.json()["response"])
```

#### Using `gpt-5-mini`

```python
import os
import requests

response = requests.post(
    "http://localhost:8000/chat",
    headers={"Authorization": f"Bearer {os.environ['WRAPPER_API_KEY']}"},
    json={
        "prompt": "Explain the difference between a list and a tuple",
        "model": "gpt-5-mini",
    },
)
print(response.json()["response"])
```

#### Streaming with `requests`

```python
import os
import requests

with requests.post(
    "http://localhost:8000/chat/stream",
    headers={"Authorization": f"Bearer {os.environ['WRAPPER_API_KEY']}"},
    json={"prompt": "Write a haiku about Python", "model": "gpt-6-luna"},
    stream=True,
) as r:
    for line in r.iter_lines():
        if line:
            text = line.decode().removeprefix("data: ")
            if text == "[DONE]":
                break
            print(text, end="", flush=True)
print()
```

---

### Python (async with `httpx`)

```python
import asyncio
import os
import httpx

HEADERS = {"Authorization": f"Bearer {os.environ['WRAPPER_API_KEY']}"}


async def ask(prompt: str, model: str = "gpt-6-luna") -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "http://localhost:8000/chat",
            headers=HEADERS,
            json={"prompt": prompt, "model": model},
        )
        r.raise_for_status()
        return r.json()["response"]


async def stream(prompt: str, model: str = "gpt-5-mini") -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/chat/stream",
            headers=HEADERS,
            json={"prompt": prompt, "model": model},
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    print(chunk, end="", flush=True)
    print()


asyncio.run(ask("What is a decorator in Python?"))
asyncio.run(stream("Count to 5 and explain each number"))
```

---

### JavaScript / Node.js (`fetch`)

#### Blocking request

```js
const res = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${process.env.WRAPPER_API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    prompt: "What is the event loop in JavaScript?",
    model: "gpt-6-luna",
  }),
});
const data = await res.json();
console.log(data.response);
```

#### Streaming with `fetch`

```js
// POST streaming via fetch + ReadableStream
const response = await fetch("http://localhost:8000/chat/stream", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${process.env.WRAPPER_API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ prompt: "Tell me a short joke", model: "gpt-5-mini" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const chunk = line.slice(6);
      if (chunk === "[DONE]") break;
      process.stdout.write(chunk);
    }
  }
}
```

---

## Tests

The tests use a fake SDK client: they don't start the Copilot runtime or spend credits.

```bash
pip install --require-hashes -r requirements-dev.lock
python -m pytest
ruff check .
pip-audit --require-hashes --disable-pip -r requirements.lock
pip-audit --require-hashes --disable-pip -r requirements-dev.lock
```

GitHub Actions runs the same checks on every pull request and push to `main`, with the tests on Python 3.11 and 3.13.
The workflow uses no secrets and makes no Copilot calls.

---

## Updating dependencies

Edit `requirements.txt` or `requirements-dev.txt`, regenerate both locks with [uv](https://docs.astral.sh/uv/), then run
the tests. Review which packages changed before committing. To move to newer versions of everything, add `--upgrade`
to both commands.

```bash
uv pip compile requirements.txt --universal --python-version 3.11 --generate-hashes -o requirements.lock
uv pip compile requirements-dev.txt -c requirements.lock --universal --python-version 3.11 --generate-hashes -o requirements-dev.lock
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `gh: command not found` | Install GitHub CLI: `brew install gh` |
| `RuntimeError: WRAPPER_API_KEY must be set` | Generate and export a key: see [Running the Server](#running-the-server) |
| `401 Missing or invalid API key` | Send `Authorization: Bearer <key>` with the key the server was started with |
| `400 Invalid host header` | Call the server as `localhost` or `127.0.0.1`, or add the hostname to `WRAPPER_ALLOWED_HOSTS` |
| First start is slow | The SDK is downloading the Copilot CLI runtime once; later starts use the cache |
| `503 Copilot client not ready` | The server is still starting up — wait a moment and retry |
| `503 Could not fetch the Copilot model list` | Check the network and `gh auth login`, then retry; the server log has the cause |
| `422` with `valid_models` | The model ID isn't available to your account: use one from `detail.valid_models` or `GET /models` |
| `ModuleNotFoundError: No module named 'copilot'` | Run `pip install --require-hashes -r requirements.lock` |
| `ImportError: cannot import name 'PermissionDecisionReject'` | An old SDK is installed: activate `.venv` (or `pip install --require-hashes -r requirements.lock` in it) |
| Startup fails with `IncompleteRead` while downloading the CLI runtime | Network dropped during the one-time 81 MB download: start again |
| Server starts but returns errors | Make sure `gh auth login` has been completed and your account has an active Copilot subscription |
