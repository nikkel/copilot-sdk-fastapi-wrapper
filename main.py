"""
FastAPI wrapper around the github-copilot-sdk.

Prerequisites:
  1. Install GitHub CLI:     brew install gh
  2. Authenticate:           gh auth login
  3. Install Copilot CLI:    gh extension install github/gh-copilot
  4. Install dependencies:   pip install -r requirements.txt
  5. Run the server:         uvicorn main:app --reload
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from copilot import CopilotClient
from copilot.session import PermissionHandler


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-4.1"
    reasoning_effort: str | None = None  # "low" | "medium" | "high" | "xhigh"


class ChatResponse(BaseModel):
    response: str
    model: str


# ---------------------------------------------------------------------------
# Application lifespan – single persistent CopilotClient
# ---------------------------------------------------------------------------

copilot_client: CopilotClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global copilot_client
    copilot_client = CopilotClient()
    await copilot_client.start()
    yield
    await copilot_client.stop()


app = FastAPI(
    title="Copilot Prompt API",
    description="Send prompts to GitHub Copilot via the github-copilot-sdk.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _send_prompt(prompt: str, model: str, reasoning_effort: str | None) -> str:
    """Open a single-use session, send *prompt*, collect the full reply."""
    if copilot_client is None:
        raise HTTPException(status_code=503, detail="Copilot client not ready")

    session_kwargs: dict = {
        "on_permission_request": PermissionHandler.approve_all,
        "model": model,
    }
    if reasoning_effort:
        session_kwargs["reasoning_effort"] = reasoning_effort

    reply_parts: list[str] = []
    done = asyncio.Event()

    async with await copilot_client.create_session(**session_kwargs) as session:

        def on_event(event) -> None:
            match event.type.value:
                case "assistant.message":
                    reply_parts.append(event.data.content or "")
                case "session.idle":
                    done.set()

        session.on(on_event)
        await session.send(prompt)
        await done.wait()

    return "".join(reply_parts)


async def _stream_prompt(
    prompt: str, model: str, reasoning_effort: str | None
) -> AsyncGenerator[str, None]:
    """Open a single-use session and yield response chunks as SSE lines."""
    if copilot_client is None:
        yield "data: error: Copilot client not ready\n\n"
        return

    session_kwargs: dict = {
        "on_permission_request": PermissionHandler.approve_all,
        "model": model,
        "streaming": True,
    }
    if reasoning_effort:
        session_kwargs["reasoning_effort"] = reasoning_effort

    done = asyncio.Event()
    chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async with await copilot_client.create_session(**session_kwargs) as session:

        def on_event(event) -> None:
            match event.type.value:
                case "assistant.message_delta":
                    delta = event.data.delta_content or ""
                    if delta:
                        chunk_queue.put_nowait(delta)
                case "session.idle":
                    chunk_queue.put_nowait(None)  # sentinel
                    done.set()

        session.on(on_event)
        await session.send(prompt)

        while True:
            chunk = await chunk_queue.get()
            if chunk is None:
                break
            # Server-Sent Event format
            yield f"data: {chunk}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, summary="Send a prompt (blocking)")
async def chat(req: PromptRequest) -> ChatResponse:
    """
    Send a prompt to GitHub Copilot and receive the complete response.
    Blocks until the model finishes generating.
    """
    response = await _send_prompt(req.prompt, req.model, req.reasoning_effort)
    return ChatResponse(response=response, model=req.model)


@app.post("/chat/stream", summary="Send a prompt (streaming SSE)")
async def chat_stream(req: PromptRequest) -> StreamingResponse:
    """
    Send a prompt and receive the response as a stream of Server-Sent Events.

    Each event has the form `data: <chunk>`.
    The final event is `data: [DONE]`.
    """
    return StreamingResponse(
        _stream_prompt(req.prompt, req.model, req.reasoning_effort),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/models", summary="List available Copilot models")
async def list_models() -> dict:
    """Return the models advertised by the connected Copilot CLI."""
    if copilot_client is None:
        raise HTTPException(status_code=503, detail="Copilot client not ready")
    models = await copilot_client.list_models()
    return {"models": models}


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}
