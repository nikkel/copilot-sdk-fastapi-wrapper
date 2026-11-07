"""A model turn must end within a time limit, however the runtime or upstream stalls.

Otherwise a stalled turn holds a session and a connection open until the server is restarted.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import API_KEY, event
from fastapi.testclient import TestClient

import main

LIMIT = 0.05
TIMEOUT_MESSAGE = "Copilot did not finish in time"
TIMEOUT_FRAME = f"event: error\ndata: {json.dumps({'message': TIMEOUT_MESSAGE})}\n\n"
BLOCKING_ROUTES = ["/chat", "/agent/step"]
ALL_ROUTES = [*BLOCKING_ROUTES, "/chat/stream"]
TOOLS = [{"type": "function", "function": {"name": "search"}}]


@pytest.fixture(autouse=True)
def short_limits(monkeypatch):
    # Through the environment, because the app's startup loads the limit from there.
    monkeypatch.setenv("COPILOT_REQUEST_TIMEOUT", str(LIMIT))
    monkeypatch.setattr(main, "_request_timeout_seconds", LIMIT)
    monkeypatch.setattr(main, "SESSION_CLOSE_TIMEOUT_SECONDS", LIMIT, raising=False)


def request(client, path, **body):
    """Sends through the ASGI app with a time limit, so a turn that never ends fails the test instead of hanging it."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        headers = {"Authorization": f"Bearer {API_KEY}"}
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost", headers=headers) as http:
            return await asyncio.wait_for(http.post(path, json={"prompt": "hi", **body}), timeout=2)

    return client.portal.call(send)


def never_idle(session):
    """The prompt is accepted, then nothing else happens."""


def record_call(session):
    session.kwargs["tools"][0].handler(SimpleNamespace(arguments={"query": "x"}, tool_name="search"))


def assert_timed_out(response) -> None:
    assert response.status_code == 504
    assert response.json() == {"detail": TIMEOUT_MESSAGE}


def idle_after(delay):
    def turn(session):
        session.emit(event("assistant.message" if not session.streaming else "assistant.message_delta",
                           content="reply", delta_content="reply"))
        asyncio.get_running_loop().call_later(delay, session.emit, event("session.idle"))

    return turn


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_timeout_log_has_no_prompt_text(client, fake_copilot, caplog, path):
    fake_copilot.turn = never_idle

    request(client, path, prompt="private-prompt-text")

    assert "timed out" in caplog.text
    assert "private-prompt-text" not in caplog.text


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_turn_finishing_just_before_the_limit_succeeds(client, fake_copilot, monkeypatch, path):
    monkeypatch.setattr(main, "_request_timeout_seconds", 0.3)
    fake_copilot.turn = idle_after(0.2)

    response = request(client, path)

    assert response.status_code == 200
    assert "reply" in response.text
    assert "event: error" not in response.text
    assert fake_copilot.turns_aborted == 0
    assert fake_copilot.sessions_closed == 1


# --- Blocking routes ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_turn_that_never_goes_idle_returns_504_and_aborts_and_closes_the_session(client, fake_copilot, caplog, path):
    fake_copilot.turn = never_idle

    assert_timed_out(request(client, path))

    assert fake_copilot.turns_aborted == 1
    assert fake_copilot.sessions_closed == 1
    assert "timed out" in caplog.text


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_stalled_send_returns_504_and_closes_the_session(client, fake_copilot, path):
    fake_copilot.stalled_calls = {"send"}

    assert_timed_out(request(client, path))

    assert fake_copilot.turns_aborted == 1
    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_stalled_create_session_returns_504(client, fake_copilot, path):
    fake_copilot.stalled_calls = {"create_session"}

    assert_timed_out(request(client, path))


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_text_before_a_timeout_is_not_returned(client, fake_copilot, path):
    fake_copilot.turn = lambda session: session.emit(event("assistant.message", content="cut off"))

    response = request(client, path)

    assert_timed_out(response)
    assert "cut off" not in response.text


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_stalled_abort_still_closes_the_session(client, fake_copilot, path):
    fake_copilot.turn = never_idle
    fake_copilot.stalled_calls = {"abort"}

    assert_timed_out(request(client, path))

    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_stalled_close_does_not_hold_the_response(client, fake_copilot, caplog, path):
    fake_copilot.turn = never_idle
    fake_copilot.stalled_calls = {"disconnect"}

    assert_timed_out(request(client, path))

    assert "Could not close" in caplog.text


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_finished_turn_is_not_aborted_and_is_closed(client, fake_copilot, path):
    assert request(client, path).status_code == 200

    assert fake_copilot.turns_aborted == 0
    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_close_failure_after_a_finished_turn_keeps_the_reply(client, fake_copilot, caplog, path):
    fake_copilot.disconnect_error = RuntimeError("close failed")

    assert request(client, path).status_code == 200

    assert "Could not close" in caplog.text


def test_agent_step_timeout_after_a_tool_call_returns_it_marked_incomplete(client, fake_copilot, caplog):
    fake_copilot.turn = record_call

    response = request(client, "/agent/step", tools=TOOLS)

    assert response.status_code == 200
    assert response.json()["tool_calls"] == [{"name": "search", "arguments": {"query": "x"}}]
    assert response.json()["incomplete"] is True
    assert fake_copilot.turns_aborted == 1
    assert fake_copilot.sessions_closed == 1
    assert "timed out" in caplog.text


# --- Streaming ----------------------------------------------------------------------------------------------------

def test_stream_that_never_goes_idle_ends_with_the_error_event(client, fake_copilot, caplog):
    fake_copilot.turn = never_idle

    response = request(client, "/chat/stream")

    assert response.status_code == 200
    assert response.text == TIMEOUT_FRAME
    assert fake_copilot.turns_aborted == 1
    assert fake_copilot.sessions_closed == 1
    assert "timed out" in caplog.text


def test_stream_keeps_chunks_sent_before_a_timeout(client, fake_copilot):
    fake_copilot.turn = lambda session: session.emit(event("assistant.message_delta", delta_content="part"))

    assert request(client, "/chat/stream").text == 'data: {"delta": "part"}\n\n' + TIMEOUT_FRAME


@pytest.mark.parametrize("call", ["send", "create_session"])
def test_stalled_sdk_call_ends_the_stream_with_the_error_event(client, fake_copilot, call):
    fake_copilot.stalled_calls = {call}

    assert request(client, "/chat/stream").text == TIMEOUT_FRAME


def test_stalled_close_does_not_hold_the_stream(client, fake_copilot):
    fake_copilot.turn = never_idle
    fake_copilot.stalled_calls = {"disconnect"}

    assert request(client, "/chat/stream").text == TIMEOUT_FRAME


def test_limit_covers_the_whole_stream_not_each_chunk(client, fake_copilot):
    """A turn that keeps producing text forever must still end, or it would never stop spending."""

    def turn(session):
        loop = asyncio.get_running_loop()

        def drip() -> None:
            session.emit(event("assistant.message_delta", delta_content="x"))
            loop.call_later(LIMIT / 5, drip)

        drip()

    fake_copilot.turn = turn

    response = request(client, "/chat/stream")

    assert response.text.endswith(TIMEOUT_FRAME)
    assert "[DONE]" not in response.text


def test_stream_closed_by_the_client_closes_the_session_without_an_abort(client, fake_copilot):
    fake_copilot.turn = lambda session: session.emit(event("assistant.message_delta", delta_content="part"))

    async def read_one_chunk_then_leave() -> str:
        frames = main._stream_prompt("hi", "gpt-6-luna", None)
        first = await anext(frames)
        await frames.aclose()
        return first

    assert client.portal.call(read_one_chunk_then_leave) == 'data: {"delta": "part"}\n\n'
    assert fake_copilot.turns_aborted == 0
    assert fake_copilot.sessions_closed == 1


def test_finished_stream_is_not_aborted_and_is_closed(client, fake_copilot):
    assert request(client, "/chat/stream").text.endswith("data: [DONE]\n\n")

    assert fake_copilot.turns_aborted == 0
    assert fake_copilot.sessions_closed == 1


# --- Setting the limit --------------------------------------------------------------------------------------------

def test_default_limit_is_two_minutes(monkeypatch):
    monkeypatch.delenv("COPILOT_REQUEST_TIMEOUT", raising=False)
    assert main._load_request_timeout() == 120


def test_limit_is_read_from_the_environment_at_startup(monkeypatch):
    monkeypatch.setenv("COPILOT_REQUEST_TIMEOUT", "12.5")

    with TestClient(main.app):
        assert main._request_timeout_seconds == 12.5


@pytest.mark.parametrize("value", ["", "abc", "0", "-1", "nan", "inf", "-inf"])
def test_server_refuses_to_start_with_an_invalid_limit(fake_copilot, monkeypatch, value):
    monkeypatch.setenv("COPILOT_REQUEST_TIMEOUT", value)

    with pytest.raises(RuntimeError, match="COPILOT_REQUEST_TIMEOUT"), TestClient(main.app):
        pass

    assert not fake_copilot.started
