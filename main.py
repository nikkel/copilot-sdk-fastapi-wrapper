"""
FastAPI wrapper around the github-copilot-sdk.

Prerequisites:
  1. Install GitHub CLI:     brew install gh
  2. Authenticate:           gh auth login
  3. Install dependencies:   pip install --require-hashes -r requirements.lock
     (github-copilot-sdk 1.x downloads the Copilot CLI runtime itself on first start)
  4. Set an API key:         export WRAPPER_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  5. Run the server:         uvicorn main:app --reload --timeout-graceful-shutdown 10
"""

import asyncio
import json
import logging
import math
import os
import re
import secrets
import unicodedata
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from difflib import get_close_matches
from time import monotonic
from typing import Annotated, Literal

from copilot import CopilotClient
from copilot.generated.rpc import ModelsListRequest, PermissionDecisionReject
from copilot.session import ReasoningEffort
from copilot.tools import Tool, ToolInvocation, ToolResult
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import AfterValidator, BaseModel, field_validator
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

# Generous defaults: each is there to stop abuse, not to shape normal use. Startup replaces them from the environment.
DEFAULT_MAX_PROMPT_CHARS = 1_000_000
DEFAULT_MAX_SYSTEM_CHARS = 1_000_000
DEFAULT_MAX_TOOLS = 128
DEFAULT_MAX_CONCURRENT_SESSIONS = 20

_max_prompt_chars = DEFAULT_MAX_PROMPT_CHARS
_max_system_chars = DEFAULT_MAX_SYSTEM_CHARS
_max_tools = DEFAULT_MAX_TOOLS
_max_concurrent_sessions = DEFAULT_MAX_CONCURRENT_SESSIONS


def _prompt_within_limit(prompt: str) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > _max_prompt_chars:
        raise ValueError(f"prompt must be at most {_max_prompt_chars} characters")
    return prompt


def _system_within_limit(system: str | None) -> str | None:
    if system is not None and len(system) > _max_system_chars:
        raise ValueError(f"system must be at most {_max_system_chars} characters")
    return system


def _not_a_slash_command(prompt: str) -> str:
    """Prompts can carry untrusted text, and the runtime has commands such as /allow-all. Whether session.send would
    act on one is untested, so a prompt that could be read as a command never reaches a session."""
    for char in prompt:
        if char.isspace() or unicodedata.category(char) == "Cf":  # Cf: zero-width characters, byte order mark, ...
            continue
        if char == "/":
            raise ValueError(
                "prompt must not start with /, ignoring leading whitespace and invisible characters, "
                "because the Copilot runtime may read it as a slash command"
            )
        break
    return prompt


Prompt = Annotated[str, AfterValidator(_prompt_within_limit), AfterValidator(_not_a_slash_command)]
SystemPrompt = Annotated[str | None, AfterValidator(_system_within_limit)]


class PromptRequest(BaseModel):
    prompt: Prompt
    model: str = "gpt-6-luna"  # cheapest non-Claude model on Copilot (2026-10)
    reasoning_effort: ReasoningEffort | None = None
    # Replaces Copilot's built-in (coding-agent) system prompt entirely: fewer tokens, and the model follows
    # your instructions instead of acting as a coding assistant.
    system: SystemPrompt = None


class ChatResponse(BaseModel):
    response: str
    model: str


# Tool names go into the session's tool allowlist, where the runtime reads entries like "builtin:*" as filter
# patterns. Without ":" or "*", a caller's name can only ever name a tool, not allow Copilot's built-in ones.
TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")


class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: dict | None = None  # a JSON schema object

    @field_validator("name")
    @classmethod
    def _name_is_plain(cls, name: str) -> str:
        if not TOOL_NAME.fullmatch(name):
            raise ValueError("name must be 1-64 letters, digits, underscores or hyphens")
        return name


class ToolSpec(BaseModel):
    """A tool the CALLER implements (OpenAI/Ollama function format: {"type": "function", "function": {...}})."""
    type: Literal["function"] = "function"
    function: ToolFunction


class AgentStepRequest(BaseModel):
    prompt: Prompt
    system: SystemPrompt = None
    tools: list[ToolSpec] = []
    model: str = "gpt-6-luna"
    reasoning_effort: ReasoningEffort | None = None

    @field_validator("tools")
    @classmethod
    def _tools_are_valid(cls, tools: list[ToolSpec]) -> list[ToolSpec]:
        if len(tools) > _max_tools:
            raise ValueError(f"tools must have at most {_max_tools} entries")
        names = [tool.function.name for tool in tools]
        if len(set(names)) != len(names):
            raise ValueError("tools must not have duplicate names")
        return tools


class ToolCall(BaseModel):
    name: str
    arguments: dict


class AgentStepResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall]
    model: str
    incomplete: bool = False  # the turn failed after recording tool_calls, so treat it as failed


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

API_KEY_ENV = "WRAPPER_API_KEY"
ALLOWED_HOSTS_ENV = "WRAPPER_ALLOWED_HOSTS"
DEFAULT_ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
OPEN_PATHS = frozenset({"/health"})

_api_key: bytes | None = None  # stays None until startup loads it, and then every request is refused


def _load_api_key() -> bytes:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} must be set. Generate one with: "
            "python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    return key.encode()


def _allowed_hosts() -> list[str]:
    hosts = [host.strip() for host in os.environ.get(ALLOWED_HOSTS_ENV, "").split(",") if host.strip()]
    # A wildcard would let a web page reach this server through DNS rebinding, which is what the allowlist stops.
    if any("*" in host for host in hosts):
        raise RuntimeError(f"{ALLOWED_HOSTS_ENV} must list exact hostnames; wildcards are not allowed")
    return hosts or DEFAULT_ALLOWED_HOSTS


def _has_valid_api_key(headers: Headers) -> bool:
    scheme, _, token = headers.get("authorization", "").partition(" ")
    if _api_key is None or scheme.lower() != "bearer":
        return False
    # Starlette decodes headers as latin-1, so this restores the bytes that were sent. compare_digest raises on a
    # non-ASCII str, which would turn a bad key into a 500.
    return secrets.compare_digest(token.encode("latin-1"), _api_key)


class RequireApiKey:
    """Refuses any request without the key unless its path is in OPEN_PATHS.

    Middleware rather than a route dependency, because /docs, /redoc and /openapi.json aren't covered by app-level
    dependencies, and so that a route added later is protected by default.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] in ("http", "websocket")
            and scope["path"] not in OPEN_PATHS
            and not _has_valid_api_key(Headers(scope=scope))
        ):
            response = JSONResponse(
                {"detail": "Missing or invalid API key"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Time limits
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_ENV = "COPILOT_REQUEST_TIMEOUT"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
SESSION_CLOSE_TIMEOUT_SECONDS = 5.0  # the SDK's abort and close calls have no time limit of their own

_request_timeout_seconds = DEFAULT_REQUEST_TIMEOUT_SECONDS


def _load_request_timeout() -> float:
    value = os.environ.get(REQUEST_TIMEOUT_ENV)
    if value is None:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        seconds = float(value)
    except ValueError:
        seconds = 0.0
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"{REQUEST_TIMEOUT_ENV} must be a positive number of seconds")
    return seconds


def _load_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number <= 0:
        raise RuntimeError(f"{name} must be a positive whole number")
    return number


# ---------------------------------------------------------------------------
# Application lifespan – single persistent CopilotClient
# ---------------------------------------------------------------------------

copilot_client: CopilotClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global copilot_client, _api_key, _request_timeout_seconds
    global _max_prompt_chars, _max_system_chars, _max_tools, _max_concurrent_sessions, _keep_sessions
    _api_key = _load_api_key()  # first, so a missing key stops startup before the runtime is launched
    _request_timeout_seconds = _load_request_timeout()
    _max_prompt_chars = _load_positive_int("COPILOT_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS)
    _max_system_chars = _load_positive_int("COPILOT_MAX_SYSTEM_CHARS", DEFAULT_MAX_SYSTEM_CHARS)
    _max_tools = _load_positive_int("COPILOT_MAX_TOOLS", DEFAULT_MAX_TOOLS)
    _max_concurrent_sessions = _load_positive_int("COPILOT_MAX_CONCURRENT_SESSIONS", DEFAULT_MAX_CONCURRENT_SESSIONS)
    _keep_sessions = _load_keep_sessions()
    # "empty" mode: the runtime adds no workspace/environment context or custom instructions to prompts
    # (fewer tokens billed, nothing about this machine is sent). It needs its own data directory.
    home = os.environ.get("COPILOT_WRAPPER_HOME", os.path.expanduser("~/.copilot-fastapi-wrapper"))
    os.makedirs(home, exist_ok=True)
    # The runtime handles untrusted prompt text and has no need for the key.
    runtime_env = {name: value for name, value in os.environ.items() if name != API_KEY_ENV}
    copilot_client = CopilotClient(mode="empty", base_directory=home, env=runtime_env)
    await copilot_client.start()
    _reset_model_cache()
    yield
    # uvicorn doesn't wait for the turns it cancels at its graceful shutdown limit, so finish them before stopping.
    await asyncio.gather(*(_close_session(session, abort=True) for session in list(_open_sessions)))
    await asyncio.gather(*_closing_sessions)
    await copilot_client.stop()


app = FastAPI(
    title="Copilot Prompt API",
    description="Send prompts to GitHub Copilot via the github-copilot-sdk.",
    version="1.0.0",
    lifespan=lifespan,
)
# The last one added runs first, so a foreign Host is refused before the key is looked at.
app.add_middleware(RequireApiKey)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts())


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

# Copilot doesn't reject an unknown model: it silently answers with a different one, which can cost far more
# than the caller chose. So only models the account lists may open a session.
MODEL_CACHE_SECONDS = 300
MODEL_LIST_TIMEOUT_SECONDS = 10
MAX_VALID_MODELS_SHOWN = 10

_model_ids: frozenset[str] = frozenset()
_model_ids_fetched_at: float | None = None
_model_ids_lock = asyncio.Lock()


def _reset_model_cache() -> None:
    global _model_ids, _model_ids_fetched_at, _model_ids_lock
    _model_ids = frozenset()
    _model_ids_fetched_at = None
    _model_ids_lock = asyncio.Lock()


async def _known_model_ids() -> frozenset[str]:
    """The account's model IDs, refetched every MODEL_CACHE_SECONDS. Raises 503 rather than using an expired list."""
    global _model_ids, _model_ids_fetched_at
    if copilot_client is None:
        raise HTTPException(status_code=503, detail="Copilot client not ready")

    async with _model_ids_lock:
        if _model_ids_fetched_at is not None and monotonic() - _model_ids_fetched_at < MODEL_CACHE_SECONDS:
            return _model_ids
        try:
            # Not client.list_models(): the SDK caches that result until the client disconnects, so it never refreshes.
            result = await copilot_client.rpc.models.list(ModelsListRequest(), timeout=MODEL_LIST_TIMEOUT_SECONDS)
            model_ids = frozenset(model.id for model in result.models)
        except Exception as exc:
            logger.warning("Could not fetch the Copilot model list: %r", exc)
            raise HTTPException(status_code=503, detail="Could not fetch the Copilot model list") from exc
        if not model_ids:
            logger.warning("Copilot returned an empty model list")
            raise HTTPException(status_code=503, detail="Could not fetch the Copilot model list")
        _model_ids, _model_ids_fetched_at = model_ids, monotonic()
        return model_ids


async def _require_known_model(model: str) -> None:
    """Call before opening a session. Raises 422 for a model the account doesn't list."""
    model_ids = await _known_model_ids()
    if model in model_ids:
        return
    # Closest IDs first, so a typo's correction isn't cut off. The slice bounds the comparison for very long input.
    closest = get_close_matches(model[:100], model_ids, n=MAX_VALID_MODELS_SHOWN)
    others = sorted(model_ids.difference(closest))
    raise HTTPException(
        status_code=422,
        detail={
            "message": "Unknown model. GET /models lists every model available to this account.",
            "valid_models": (closest + others)[:MAX_VALID_MODELS_SHOWN],
        },
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


# The runtime's error text stays in the server log, because it can hold internal details.
UPSTREAM_FAILURE_MESSAGE = "Copilot failed to complete the request"
EMPTY_REPLY = "empty reply"
TIMEOUT_MESSAGE = "Copilot did not finish in time"


def _session_error_text(event) -> str:
    return getattr(event.data, "message", "") or "unknown error"


def _log_upstream_failure(detail: object) -> None:
    logger.warning("Copilot session failed: %r", detail)


def _upstream_failure(detail: object) -> HTTPException:
    _log_upstream_failure(detail)
    return HTTPException(status_code=502, detail=UPSTREAM_FAILURE_MESSAGE)


def _log_turn_timeout() -> None:
    logger.warning("Copilot turn timed out after %s seconds", _request_timeout_seconds)


def _turn_timed_out() -> HTTPException:
    _log_turn_timeout()
    return HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)


def _sse_error(message: str) -> str:
    return f"event: error\ndata: {json.dumps({'message': message})}\n\n"


def _session_kwargs(model: str, reasoning_effort: str | None, system: str | None, **extra) -> dict:
    kwargs: dict = {**SAFE_SESSION, "model": model, **extra}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if system:
        kwargs["system_message"] = {"mode": "replace", "content": system}
    return kwargs


def _turn_deadline() -> float:
    """One time limit for a whole turn. Each wait checks this deadline, because a timeout scope can't stay open across
    the stream's yields: it would fire in whatever the consumer is doing at the time, not in the turn."""
    return asyncio.get_running_loop().time() + _request_timeout_seconds


async def _end_session_call(call) -> None:
    try:
        async with asyncio.timeout(SESSION_CLOSE_TIMEOUT_SECONDS):
            await call()
    except Exception:
        logger.warning("Could not close the Copilot session cleanly", exc_info=True)


KEEP_SESSIONS_ENV = "COPILOT_KEEP_SESSIONS"

_keep_sessions = False
_open_sessions: set = set()
_closing_sessions: set[asyncio.Task] = set()  # keeps a reference, so a running close isn't garbage collected


def _load_keep_sessions() -> bool:
    value = os.environ.get(KEEP_SESSIONS_ENV, "false")
    if value not in ("true", "false"):
        raise RuntimeError(f"{KEEP_SESSIONS_ENV} must be true or false")
    if value == "true":
        logger.warning("%s is true, so sessions stay on disk with their prompts and replies", KEEP_SESSIONS_ENV)
    return value == "true"


async def _delete_session(session_id: str) -> None:
    """Only ever this request's session, by ID. A failure logs the exception's type alone, so no prompt text can reach
    the log, and doesn't change the caller's response."""
    try:
        async with asyncio.timeout(SESSION_CLOSE_TIMEOUT_SECONDS):
            await copilot_client.delete_session(session_id)
    except Exception as exc:  # noqa: BLE001 - any failure must leave the response alone, and exc's text isn't logged
        logger.warning("Could not delete Copilot session %s: %s", session_id, type(exc).__name__)


async def _close_session(session, abort: bool) -> None:
    if session not in _open_sessions:
        return  # shutdown and the request's own clean-up can both get here
    _open_sessions.discard(session)

    async def close() -> None:
        if abort:
            await _end_session_call(session.abort)
        await _end_session_call(session.disconnect)
        if not _keep_sessions:
            await _delete_session(session.session_id)

    # In its own task, so it finishes even if this request is cancelled. On a client disconnect, Starlette cancels a
    # stream again at every await, which would interrupt the SDK's calls here and leave the session open.
    closing = asyncio.create_task(close())
    _closing_sessions.add(closing)
    closing.add_done_callback(_closing_sessions.discard)
    await asyncio.shield(closing)


@asynccontextmanager
async def _turn_session(session_kwargs: dict, deadline: float) -> AsyncIterator:
    """A session that is always closed, without letting a stalled runtime hold the caller. A turn that timed out or that
    nobody is waiting for is aborted first, because closing keeps the session on the runtime and may not stop a turn
    that's still spending."""
    async with asyncio.timeout_at(deadline):
        session = await copilot_client.create_session(**session_kwargs)
    _open_sessions.add(session)
    abort = False
    try:
        yield session
    except (TimeoutError, asyncio.CancelledError, GeneratorExit):
        abort = True
        raise
    finally:
        await _close_session(session, abort)


async def _until_client_disconnects(request: Request, turn: Coroutine):
    """Runs a blocking route's turn, and cancels it if the client leaves first."""

    async def client_left() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    turn_task = asyncio.create_task(turn)
    disconnect_task = asyncio.create_task(client_left())
    try:
        await asyncio.wait({turn_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
        if turn_task.done():
            return turn_task.result()
        logger.info("Client disconnected, so the Copilot turn was cancelled")
        raise HTTPException(status_code=499, detail="Client disconnected")
    finally:
        disconnect_task.cancel()
        if not turn_task.done():
            turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)


SESSION_RETRY_AFTER_SECONDS = 5

_sessions_in_use = 0


def _take_session_slot() -> None:
    """Call before opening a session, and pair with _free_session_slot. Raises 429 when every slot is taken."""
    global _sessions_in_use
    if _sessions_in_use >= _max_concurrent_sessions:
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent sessions",
            headers={"Retry-After": str(SESSION_RETRY_AFTER_SECONDS)},
        )
    _sessions_in_use += 1


def _free_session_slot() -> None:
    global _sessions_in_use
    _sessions_in_use -= 1


class _SessionStreamingResponse(StreamingResponse):
    """Holds a session slot for the life of the stream, including when the client leaves before the stream starts."""

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.body_iterator.aclose()  # a stream cancelled at a yield would otherwise wait for collection
            finally:
                _free_session_slot()


async def _send_prompt(prompt: str, model: str, reasoning_effort: str | None, system: str | None = None) -> str:
    """Open a single-use session, send *prompt*, collect the full reply."""
    if copilot_client is None:
        raise HTTPException(status_code=503, detail="Copilot client not ready")

    session_kwargs = _session_kwargs(model, reasoning_effort, system)

    reply_parts: list[str] = []
    errors: list[str] = []
    done = asyncio.Event()
    deadline = _turn_deadline()

    try:
        async with _turn_session(session_kwargs, deadline) as session:

            def on_event(event) -> None:
                match event.type.value:
                    case "assistant.message":
                        reply_parts.append(event.data.content or "")
                    case "session.error":
                        errors.append(_session_error_text(event))
                        done.set()  # idle may not follow an error
                    case "session.idle":
                        done.set()

            session.on(on_event)
            async with asyncio.timeout_at(deadline):
                await session.send(prompt)
                await done.wait()
    except TimeoutError as exc:
        raise _turn_timed_out() from exc
    except Exception as exc:
        raise _upstream_failure(exc) from exc

    if errors:
        raise _upstream_failure(errors[-1])
    reply = "".join(reply_parts)
    if not reply.strip():
        raise _upstream_failure(EMPTY_REPLY)
    return reply


async def _stream_prompt(
    prompt: str, model: str, reasoning_effort: str | None, system: str | None = None
) -> AsyncGenerator[str, None]:
    """Open a single-use session and yield response chunks as SSE lines."""
    if copilot_client is None:
        yield _sse_error("Copilot client not ready")
        return

    session_kwargs = _session_kwargs(model, reasoning_effort, system, streaming=True)

    done = asyncio.Event()
    chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()
    errors: list[str] = []
    received_chunk = False
    deadline = _turn_deadline()

    # The 200 status has already been sent by the time anything here fails, so failures are reported in the stream.
    try:
        async with _turn_session(session_kwargs, deadline) as session:

            def on_event(event) -> None:
                match event.type.value:
                    case "assistant.message_delta":
                        delta = event.data.delta_content or ""
                        if delta:
                            chunk_queue.put_nowait(delta)
                    case "session.error":
                        errors.append(_session_error_text(event))
                        chunk_queue.put_nowait(None)  # idle may not follow an error
                    case "session.idle":
                        chunk_queue.put_nowait(None)  # sentinel
                        done.set()

            session.on(on_event)
            async with asyncio.timeout_at(deadline):
                await session.send(prompt)

            while True:
                async with asyncio.timeout_at(deadline):
                    chunk = await chunk_queue.get()
                if chunk is None:
                    break
                received_chunk = True
                # Encoded, so that text in a chunk can't forge an SSE field or a [DONE]
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
    except TimeoutError:
        _log_turn_timeout()
        yield _sse_error(TIMEOUT_MESSAGE)
        return
    except Exception:
        logger.exception("Copilot session failed")
        yield _sse_error(UPSTREAM_FAILURE_MESSAGE)
        return

    if errors or not received_chunk:
        _log_upstream_failure(errors[-1] if errors else EMPTY_REPLY)
        yield _sse_error(UPSTREAM_FAILURE_MESSAGE)
        return

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
        tools.append(Tool(name=fn.name, description=fn.description, handler=capture,
                          parameters=fn.parameters or {"type": "object", "properties": {}},
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
    failed = False
    timed_out = False
    deadline = _turn_deadline()

    try:
        async with _turn_session(session_kwargs, deadline) as session:

            def on_event(event) -> None:
                match event.type.value:
                    case "assistant.message":
                        reply_parts.append(event.data.content or "")
                    case "session.error":
                        errors.append(_session_error_text(event))
                        done.set()  # idle may not follow an error
                    case "session.idle":
                        done.set()

            session.on(on_event)
            async with asyncio.timeout_at(deadline):
                await session.send(req.prompt)
                await done.wait()
    except TimeoutError:
        timed_out = True
    except Exception:
        logger.warning("Copilot session failed", exc_info=True)
        failed = True

    if errors:
        _log_upstream_failure(errors[-1])
        failed = True
    # Recorded tool calls are returned, marked incomplete, so the caller decides what to do with them.
    if timed_out and not calls:
        raise _turn_timed_out()
    if timed_out:
        _log_turn_timeout()
        failed = True
    if failed and not calls:
        raise HTTPException(status_code=502, detail=UPSTREAM_FAILURE_MESSAGE)
    content = "".join(reply_parts).strip()
    if not content and not calls:
        raise _upstream_failure(EMPTY_REPLY)
    return AgentStepResponse(content=content, tool_calls=calls, model=req.model, incomplete=failed)


@app.post("/agent/step", response_model=AgentStepResponse, summary="One step of an agent (native tool calling)")
async def agent_step(req: AgentStepRequest, request: Request) -> AgentStepResponse:
    """
    Send a system prompt, the conversation so far and the tools YOUR app implements. The model either calls
    tools (returned in `tool_calls`, never executed here) or answers in `content`. Run the tools yourself and
    send the next step with their results in the prompt.
    """
    await _require_known_model(req.model)
    _take_session_slot()
    try:
        return await _until_client_disconnects(request, _agent_step(req))
    finally:
        _free_session_slot()


@app.post("/chat", response_model=ChatResponse, summary="Send a prompt (blocking)")
async def chat(req: PromptRequest, request: Request) -> ChatResponse:
    """
    Send a prompt to GitHub Copilot and receive the complete response.
    Blocks until the model finishes generating.
    """
    await _require_known_model(req.model)
    _take_session_slot()
    try:
        response = await _until_client_disconnects(
            request, _send_prompt(req.prompt, req.model, req.reasoning_effort, req.system)
        )
    finally:
        _free_session_slot()
    return ChatResponse(response=response, model=req.model)


@app.post("/chat/stream", summary="Send a prompt (streaming SSE)")
async def chat_stream(req: PromptRequest) -> StreamingResponse:
    """
    Send a prompt and receive the response as a stream of Server-Sent Events.

    Each event has the form `data: {"delta": "<chunk>"}`.
    A successful stream ends with `data: [DONE]`. A failed one ends with an `error` event and no `[DONE]`.
    """
    await _require_known_model(req.model)  # before the response starts, so errors are real HTTP status codes
    _take_session_slot()
    return _SessionStreamingResponse(
        _stream_prompt(req.prompt, req.model, req.reasoning_effort, req.system),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/models", summary="List available Copilot models")
async def list_models() -> dict:
    """Return the models advertised by the connected Copilot CLI."""
    if copilot_client is None:
        raise HTTPException(status_code=503, detail="Copilot client not ready")
    try:
        async with asyncio.timeout(MODEL_LIST_TIMEOUT_SECONDS):
            models = await copilot_client.list_models()
    except Exception as exc:
        logger.warning("Could not fetch the Copilot model list: %r", exc)
        raise HTTPException(status_code=503, detail="Could not fetch the Copilot model list") from exc
    return {"models": models}


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}
