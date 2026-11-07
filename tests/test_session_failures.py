"""A failed turn must reach the caller as a failure, not as an empty success or a normal end of stream.

The runtime's error text stays in the server log, because it can hold internal details.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import API_KEY, event

import main

MARKER = "internal-detail-marker"
GENERIC_MESSAGE = "Copilot failed to complete the request"
ERROR_FRAME = f"event: error\ndata: {json.dumps({'message': GENERIC_MESSAGE})}\n\n"
DONE_FRAME = "data: [DONE]\n\n"
BLOCKING_ROUTES = ["/chat", "/agent/step"]
ALL_ROUTES = [*BLOCKING_ROUTES, "/chat/stream"]


def request(client, path, **body):
    """Sends through the ASGI app with a time limit, so a turn that never ends fails the test instead of hanging it."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        headers = {"Authorization": f"Bearer {API_KEY}"}
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost", headers=headers) as http:
            return await asyncio.wait_for(http.post(path, json={"prompt": "hi", **body}), timeout=2)

    return client.portal.call(send)


def error_event():
    return event("session.error", error_type="query", message=MARKER)


def error_then_idle(session):
    session.emit(error_event(), event("session.idle"))


def error_without_idle(session):
    session.emit(error_event())


FAILED_TURNS = [pytest.param(error_then_idle, id="error-then-idle"), pytest.param(error_without_idle, id="error-only")]


def assert_generic_502(response, caplog) -> None:
    assert response.status_code == 502
    assert response.json() == {"detail": GENERIC_MESSAGE}
    assert MARKER in caplog.text


# --- Blocking routes ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", BLOCKING_ROUTES)
@pytest.mark.parametrize("turn", FAILED_TURNS)
def test_session_error_returns_a_generic_502_and_closes_the_session(client, fake_copilot, caplog, path, turn):
    fake_copilot.turn = turn

    assert_generic_502(request(client, path), caplog)
    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_text_before_a_session_error_is_not_returned_as_a_success(client, fake_copilot, caplog, path):
    fake_copilot.turn = lambda session: session.emit(event("assistant.message", content="cut off"), error_event())

    response = request(client, path)

    assert_generic_502(response, caplog)
    assert "cut off" not in response.text


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_exception_from_send_returns_a_generic_502(client, fake_copilot, caplog, path):
    fake_copilot.send_error = RuntimeError(MARKER)

    assert_generic_502(request(client, path), caplog)
    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_exception_from_create_session_returns_a_generic_502(client, fake_copilot, caplog, path):
    fake_copilot.create_session_error = RuntimeError(MARKER)

    assert_generic_502(request(client, path), caplog)


TOOLS = [{"type": "function", "function": {"name": "search"}}]


def record_call(session):
    session.kwargs["tools"][0].handler(SimpleNamespace(arguments={"query": "x"}, tool_name="search"))


def test_agent_step_session_error_after_a_tool_call_returns_it_marked_incomplete(client, fake_copilot, caplog):
    def turn(session):
        record_call(session)
        session.emit(event("assistant.message", content="cut off"), error_event())

    fake_copilot.turn = turn

    response = request(client, "/agent/step", tools=TOOLS)

    assert response.status_code == 200
    assert response.json()["tool_calls"] == [{"name": "search", "arguments": {"query": "x"}}]
    assert response.json()["incomplete"] is True
    assert MARKER not in response.text
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any(MARKER in message for message in warnings)
    assert fake_copilot.sessions_closed == 1


def test_agent_step_exception_after_a_tool_call_returns_it_marked_incomplete(client, fake_copilot, caplog):
    def turn(session):
        record_call(session)
        raise RuntimeError(MARKER)

    fake_copilot.turn = turn

    response = request(client, "/agent/step", tools=TOOLS)

    assert response.status_code == 200
    assert response.json()["tool_calls"] == [{"name": "search", "arguments": {"query": "x"}}]
    assert response.json()["incomplete"] is True
    assert MARKER not in response.text
    assert MARKER in caplog.text


def test_agent_step_answer_is_not_incomplete(client):
    response = request(client, "/agent/step")

    assert response.status_code == 200
    assert response.json()["content"] == "fake reply"
    assert response.json()["incomplete"] is False


def test_agent_step_tool_call_without_an_error_is_not_incomplete(client, fake_copilot):
    def turn(session):
        record_call(session)
        session.emit(event("session.idle"))

    fake_copilot.turn = turn

    response = request(client, "/agent/step", tools=TOOLS)

    assert response.json()["tool_calls"] == [{"name": "search", "arguments": {"query": "x"}}]
    assert response.json()["incomplete"] is False


# --- Streaming ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("turn", FAILED_TURNS)
def test_stream_with_a_session_error_ends_with_the_error_event_and_no_done(client, fake_copilot, caplog, turn):
    fake_copilot.turn = turn

    response = request(client, "/chat/stream")

    assert response.status_code == 200
    assert response.text == ERROR_FRAME
    assert MARKER in caplog.text
    assert fake_copilot.sessions_closed == 1


def test_stream_keeps_chunks_sent_before_a_session_error(client, fake_copilot):
    fake_copilot.turn = lambda session: session.emit(event("assistant.message_delta", delta_content="part"), error_event())

    assert request(client, "/chat/stream").text == 'data: {"delta": "part"}\n\n' + ERROR_FRAME


def test_exception_from_send_ends_the_stream_with_the_error_event(client, fake_copilot, caplog):
    fake_copilot.send_error = RuntimeError(MARKER)

    response = request(client, "/chat/stream")

    assert response.text == ERROR_FRAME
    assert MARKER in caplog.text
    assert fake_copilot.sessions_closed == 1


def test_exception_from_create_session_ends_the_stream_with_the_error_event(client, fake_copilot, caplog):
    fake_copilot.create_session_error = RuntimeError(MARKER)

    response = request(client, "/chat/stream")

    assert response.text == ERROR_FRAME
    assert MARKER in caplog.text


def test_stream_with_no_client_ends_with_the_error_event():
    async def frames() -> list[str]:
        return [frame async for frame in main._stream_prompt("hi", "gpt-6-luna", None)]

    frame = f"event: error\ndata: {json.dumps({'message': 'Copilot client not ready'})}\n\n"
    assert asyncio.run(frames()) == [frame]


def test_successful_stream_still_ends_with_done(client):
    assert request(client, "/chat/stream").text == 'data: {"delta": "fake reply"}\n\n' + DONE_FRAME


# --- Reasoning effort ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_ROUTES)
@pytest.mark.parametrize("effort", ["ultra-max", "", "HIGH", 1])
def test_unknown_reasoning_effort_returns_422_and_creates_no_session(client, fake_copilot, path, effort):
    response = request(client, path, reasoning_effort=effort)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "reasoning_effort"]
    assert fake_copilot.sessions == []


@pytest.mark.parametrize("path", ALL_ROUTES)
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_known_reasoning_effort_reaches_the_session(client, fake_copilot, path, effort):
    assert request(client, path, reasoning_effort=effort).status_code == 200
    assert fake_copilot.sessions[0]["reasoning_effort"] == effort


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_no_reasoning_effort_is_still_accepted(client, fake_copilot, path):
    assert request(client, path, reasoning_effort=None).status_code == 200
    assert "reasoning_effort" not in fake_copilot.sessions[0]
