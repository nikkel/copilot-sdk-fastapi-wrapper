"""
FastAPI wrapper around the github-copilot-sdk.

Prerequisites:
  1. Install GitHub CLI:     brew install gh
  2. Authenticate:           gh auth login
  3. Install dependencies:   pip install -r requirements.txt
     (github-copilot-sdk 1.x downloads the Copilot CLI runtime itself on first start)
  4. Run the server:         uvicorn main:app --reload
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from copilot import CopilotClient
from copilot.tools import Tool, ToolInvocation, ToolResult
from copilot.generated.rpc import PermissionDecisionReject


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-6-luna"  # cheapest non-Claude model on Copilot (2026-10)
    reasoning_effort: str | None = None  # "low" | "medium" | "high" | "xhigh"
    # Replaces Copilot's built-in (coding-agent) system prompt entirely: fewer tokens, and the model follows
    # your instructions instead of acting as a coding assistant.
    system: str | None = None


class ChatResponse(BaseModel):
    response: str
    model: str


class ToolSpec(BaseModel):
    """A tool the CALLER implements (OpenAI/Ollama function format: {"type": "function", "function": {...}})."""
    type: str = "function"
    function: dict


class AgentStepRequest(BaseModel):
    prompt: str
    system: str | None = None
    tools: list[ToolSpec] = []
    model: str = "gpt-6-luna"
    reasoning_effort: str | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict


class AgentStepResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall]
    model: str


# ---------------------------------------------------------------------------
# Application lifespan – single persistent CopilotClient
# ---------------------------------------------------------------------------

copilot_client: CopilotClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global copilot_client
    # "empty" mode: the runtime adds no workspace/environment context or custom instructions to prompts
    # (fewer tokens billed, nothing about this machine is sent). It needs its own data directory.
    home = os.environ.get("COPILOT_WRAPPER_HOME", os.path.expanduser("~/.copilot-fastapi-wrapper"))
    os.makedirs(home, exist_ok=True)
    copilot_client = CopilotClient(mode="empty", base_directory=home)
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

def deny_all(request, invocation) -> PermissionDecisionReject:
    """Refuse every permission request (shell, file writes, URLs, ...).

    The API only answers prompts. Prompts may contain untrusted text (e.g. web pages from a browsing agent),
    so the session must never be able to act on this machine.
    """
    return PermissionDecisionReject(feedback="This API only answers prompts; no actions are allowed.")


# Answer-only sessions: no built-in Copilot CLI tools (shell, file edits, ...) and every permission denied.
SAFE_SESSION = {"on_permission_request": deny_all, "available_tools": []}


def _session_kwargs(model: str, reasoning_effort: str | None, system: str | None, **extra) -> dict:
    kwargs: dict = {**SAFE_SESSION, "model": model, **extra}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if system:
        kwargs["system_message"] = {"mode": "replace", "content": system}
    return kwargs


async def _send_prompt(prompt: str, model: str, reasoning_effort: str | None, system: str | None = None) -> str:
    """Open a single-use session, send *prompt*, collect the full reply."""
    if copilot_client is None:
        raise HTTPException(status_code=503, detail="Copilot client not ready")

    session_kwargs = _session_kwargs(model, reasoning_effort, system)

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
    prompt: str, model: str, reasoning_effort: str | None, system: str | None = None
) -> AsyncGenerator[str, None]:
    """Open a single-use session and yield response chunks as SSE lines."""
    if copilot_client is None:
        yield "data: error: Copilot client not ready\n\n"
        return

    session_kwargs = _session_kwargs(model, reasoning_effort, system, streaming=True)

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

def _capturing_tools(specs: list[ToolSpec], calls: list[ToolCall]) -> list[Tool]:
    """The caller's tools, registered natively so the model uses real function calling. The handler only
    RECORDS the call (nothing runs here) and the tool is terminal, so the turn ends and the caller executes it."""

    def capture(invocation: ToolInvocation) -> ToolResult:
        args = invocation.arguments if isinstance(invocation.arguments, dict) else {}
        calls.append(ToolCall(name=invocation.tool_name, arguments=args))
        return ToolResult(text_result_for_llm="Handed to the app, which runs it and reports back.")

    tools = []
    for spec in specs:
        fn = spec.function
        tools.append(Tool(name=fn["name"], description=fn.get("description", ""), handler=capture,
                          parameters=fn.get("parameters") or {"type": "object", "properties": {}},
                          skip_permission=True, is_terminal=True,
                          overrides_built_in_tool=True))  # e.g. web_fetch: the caller's version wins
    return tools


async def _agent_step(req: AgentStepRequest) -> AgentStepResponse:
    """One decision of an agent whose tools run elsewhere: returns the model's text and/or its tool calls."""
    if copilot_client is None:
        raise HTTPException(status_code=503, detail="Copilot client not ready")
    calls: list[ToolCall] = []
    tools = _capturing_tools(req.tools, calls)
    session_kwargs = _session_kwargs(req.model, req.reasoning_effort, req.system, tools=tools)
    session_kwargs["available_tools"] = [t.name for t in tools]  # only the caller's tools, no built-ins
    reply_parts: list[str] = []
    errors: list[str] = []
    done = asyncio.Event()

    async with await copilot_client.create_session(**session_kwargs) as session:

        def on_event(event) -> None:
            match event.type.value:
                case "assistant.message":
                    reply_parts.append(event.data.content or "")
                case "session.error":
                    errors.append(getattr(event.data, "message", "") or "unknown error")
                case "session.idle":
                    done.set()

        session.on(on_event)
        await session.send(req.prompt)
        await done.wait()

    if errors and not calls and not "".join(reply_parts).strip():
        raise HTTPException(status_code=502, detail=f"Copilot: {errors[-1]}")
    return AgentStepResponse(content="".join(reply_parts).strip(), tool_calls=calls, model=req.model)


@app.post("/agent/step", response_model=AgentStepResponse, summary="One step of an agent (native tool calling)")
async def agent_step(req: AgentStepRequest) -> AgentStepResponse:
    """
    Send a system prompt, the conversation so far and the tools YOUR app implements. The model either calls
    tools (returned in `tool_calls`, never executed here) or answers in `content`. Run the tools yourself and
    send the next step with their results in the prompt.
    """
    return await _agent_step(req)


@app.post("/chat", response_model=ChatResponse, summary="Send a prompt (blocking)")
async def chat(req: PromptRequest) -> ChatResponse:
    """
    Send a prompt to GitHub Copilot and receive the complete response.
    Blocks until the model finishes generating.
    """
    response = await _send_prompt(req.prompt, req.model, req.reasoning_effort, req.system)
    return ChatResponse(response=response, model=req.model)


@app.post("/chat/stream", summary="Send a prompt (streaming SSE)")
async def chat_stream(req: PromptRequest) -> StreamingResponse:
    """
    Send a prompt and receive the response as a stream of Server-Sent Events.

    Each event has the form `data: <chunk>`.
    The final event is `data: [DONE]`.
    """
    return StreamingResponse(
        _stream_prompt(req.prompt, req.model, req.reasoning_effort, req.system),
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
