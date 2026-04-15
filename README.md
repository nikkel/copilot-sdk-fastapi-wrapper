# Copilot Prompt API

A FastAPI wrapper around the [github-copilot-sdk](https://pypi.org/project/github-copilot-sdk/) that exposes GitHub Copilot as a local HTTP API. Send prompts, stream responses, and list available models — all from any HTTP client.

---

## Free Models

The following models have a **0× premium-request multiplier**, meaning they are unlimited at no extra cost on any paid Copilot plan (e.g. Copilot Pro ~£10/month):

| Model ID | Best for |
|---|---|
| `gpt-4.1` | General-purpose coding, explanations, writing (**default**) |
| `gpt-5-mini` | Fast answers, lightweight coding tasks, multimodal input |

> **Copilot Free users** still get 50 premium requests/month across all models.

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

### 3 — Install the Copilot CLI extension

```bash
gh extension install github/gh-copilot
```

Verify it works:

```bash
gh copilot --version
```

### 4 — Install Python dependencies

Requires **Python 3.11+**.

```bash
pip install -r requirements.txt
```

`requirements.txt` installs:
- `github-copilot-sdk` — Python SDK that communicates with the Copilot CLI
- `fastapi` — web framework
- `uvicorn` — ASGI server

---

## Running the Server

```bash
uvicorn main:app --reload
```

The server starts on **http://localhost:8000**.

| URL | Description |
|---|---|
| http://localhost:8000/docs | **Swagger UI** — interactive API explorer |
| http://localhost:8000/redoc | ReDoc — alternative docs viewer |
| http://localhost:8000/openapi.json | Raw OpenAPI schema |

---

## Swagger UI (Interactive Docs)

Once the server is running, open **http://localhost:8000/docs** in your browser.

You will see all endpoints listed. Click any endpoint → **"Try it out"** → fill in the request body → **"Execute"** to send a real request and see the live response.

![Swagger UI is available at /docs after starting the server]

---

## API Reference

### `POST /chat` — Send a prompt (blocking)

Returns the full response once the model has finished generating.

**Request body:**

```json
{
  "prompt": "string",
  "model": "gpt-4.1",
  "reasoning_effort": null
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | ✅ | The message to send to Copilot |
| `model` | string | ❌ | Model ID (default: `gpt-4.1`) |
| `reasoning_effort` | string \| null | ❌ | `"low"` / `"medium"` / `"high"` / `"xhigh"` — only for models that support reasoning |

**Response:**

```json
{
  "response": "The assistant's full reply",
  "model": "gpt-4.1"
}
```

---

### `POST /chat/stream` — Send a prompt (streaming)

Returns the response as a stream of [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events). Each event contains one chunk of text. The final event is `data: [DONE]`.

Same request body as `/chat`.

---

### `GET /models` — List available models

Returns the models advertised by the connected Copilot CLI.

**Response:**

```json
{
  "models": ["gpt-4.1", "gpt-5-mini", "claude-sonnet-4.6", "..."]
}
```

---

### `GET /health` — Health check

```json
{ "status": "ok" }
```

---

## Examples

### curl

#### Basic prompt with the default free model (`gpt-4.1`)

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a Python generator?"}'
```

#### Explicit free model `gpt-5-mini`

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function that flattens a nested list",
    "model": "gpt-5-mini"
  }'
```

#### Streaming response with `gpt-4.1`

```bash
curl -s -N -X POST http://localhost:8000/chat/stream \
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
curl -s http://localhost:8000/models | python3 -m json.tool
```

---

### Python (`requests`)

#### Blocking request

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "prompt": "Give me a one-liner to reverse a string in Python",
        "model": "gpt-4.1",
    },
)
print(response.json()["response"])
```

#### Using `gpt-5-mini`

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "prompt": "Explain the difference between a list and a tuple",
        "model": "gpt-5-mini",
    },
)
print(response.json()["response"])
```

#### Streaming with `requests`

```python
import requests

with requests.post(
    "http://localhost:8000/chat/stream",
    json={"prompt": "Write a haiku about Python", "model": "gpt-4.1"},
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
import httpx


async def ask(prompt: str, model: str = "gpt-4.1") -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "http://localhost:8000/chat",
            json={"prompt": prompt, "model": model},
        )
        r.raise_for_status()
        return r.json()["response"]


async def stream(prompt: str, model: str = "gpt-5-mini") -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/chat/stream",
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
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "What is the event loop in JavaScript?",
    model: "gpt-4.1",
  }),
});
const data = await res.json();
console.log(data.response);
```

#### Streaming with `EventSource` (browser)

```js
// POST streaming via fetch + ReadableStream
const response = await fetch("http://localhost:8000/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
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

## Troubleshooting

| Error | Fix |
|---|---|
| `gh: command not found` | Install GitHub CLI: `brew install gh` |
| `gh copilot: command not found` | Run `gh extension install github/gh-copilot` |
| `503 Copilot client not ready` | The server is still starting up — wait a moment and retry |
| `ModuleNotFoundError: No module named 'copilot'` | Run `pip install -r requirements.txt` |
| Server starts but returns errors | Make sure `gh auth login` has been completed and your account has an active Copilot subscription |
