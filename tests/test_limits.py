"""One caller must not be able to send unbounded input or hold unbounded sessions.

Each limit refuses a request before a session opens, so a refused request spends nothing.
"""

import asyncio

import httpx
import pytest
from conftest import API_KEY, event
from fastapi.testclient import TestClient

import main

SESSION_ROUTES = ["/chat", "/chat/stream", "/agent/step"]
TOOL = {"type": "function", "function": {"name": "search"}}


def tools(count: int) -> list[dict]:
    return [{"type": "function", "function": {"name": f"tool{i}"}} for i in range(count)]


def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )


def assert_refused_on(response, field: str) -> None:
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


# --- Sizes --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_prompt_over_the_limit_returns_422_and_creates_no_session(client, fake_copilot, monkeypatch, path):
    monkeypatch.setattr(main, "_max_prompt_chars", 10)

    assert_refused_on(client.post(path, json={"prompt": "x" * 11}), "prompt")
    assert fake_copilot.sessions == []


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_prompt_at_the_limit_is_accepted(client, monkeypatch, path):
    monkeypatch.setattr(main, "_max_prompt_chars", 10)

    assert client.post(path, json={"prompt": "x" * 10}).status_code == 200


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_system_over_the_limit_returns_422_and_creates_no_session(client, fake_copilot, monkeypatch, path):
    monkeypatch.setattr(main, "_max_system_chars", 10)

    assert_refused_on(client.post(path, json={"prompt": "hi", "system": "x" * 11}), "system")
    assert fake_copilot.sessions == []


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_system_at_the_limit_or_absent_is_accepted(client, monkeypatch, path):
    monkeypatch.setattr(main, "_max_system_chars", 10)

    assert client.post(path, json={"prompt": "hi", "system": "x" * 10}).status_code == 200
    assert client.post(path, json={"prompt": "hi", "system": None}).status_code == 200


def test_too_many_tools_returns_422_and_creates_no_session(client, fake_copilot, monkeypatch):
    monkeypatch.setattr(main, "_max_tools", 2)

    assert_refused_on(client.post("/agent/step", json={"prompt": "hi", "tools": tools(3)}), "tools")
    assert fake_copilot.sessions == []


def test_tools_at_the_limit_are_accepted(client, monkeypatch):
    monkeypatch.setattr(main, "_max_tools", 2)

    assert client.post("/agent/step", json={"prompt": "hi", "tools": tools(2)}).status_code == 200


# --- Concurrent sessions ------------------------------------------------------------------------------------------

def test_requests_over_the_limit_get_429_with_retry_after_and_create_no_session(client, fake_copilot, monkeypatch):
    monkeypatch.setattr(main, "_max_concurrent_sessions", 1)
    held_sessions = []
    started = asyncio.Event()

    def hold(session):
        held_sessions.append(session)
        started.set()

    fake_copilot.turn = hold

    async def scenario() -> tuple[list[httpx.Response], httpx.Response]:
        async with http_client() as http:
            first = asyncio.create_task(http.post("/chat", json={"prompt": "hi"}))
            await asyncio.wait_for(started.wait(), timeout=2)

            refused = [await http.post(path, json={"prompt": "hi"}) for path in SESSION_ROUTES]
            assert len(fake_copilot.sessions) == 1

            fake_copilot.turn = None
            held_sessions[0].emit(event("assistant.message", content="done"), event("session.idle"))
            assert (await asyncio.wait_for(first, timeout=2)).status_code == 200

            return refused, await http.post("/chat", json={"prompt": "hi"})

    refused, afterwards = client.portal.call(scenario)

    for response in refused:
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) > 0
        assert response.json() == {"detail": "Too many concurrent sessions"}
    assert afterwards.status_code == 200
    assert len(fake_copilot.sessions) == 2


def slots_in_use() -> int:
    return main._sessions_in_use


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_finished_turn_frees_its_slot(client, path):
    assert client.post(path, json={"prompt": "hi"}).status_code == 200
    assert slots_in_use() == 0


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_failed_turn_frees_its_slot(client, fake_copilot, path):
    fake_copilot.send_error = RuntimeError("failed")

    client.post(path, json={"prompt": "hi"})

    assert slots_in_use() == 0


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_timed_out_turn_frees_its_slot(client, fake_copilot, monkeypatch, path):
    monkeypatch.setattr(main, "_request_timeout_seconds", 0.01)
    fake_copilot.turn = lambda session: None

    client.post(path, json={"prompt": "hi"})

    assert slots_in_use() == 0


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_refused_model_takes_no_slot_and_is_reported_first(client, monkeypatch, path):
    monkeypatch.setattr(main, "_max_concurrent_sessions", 0)

    assert client.post(path, json={"prompt": "hi", "model": "not-a-model"}).status_code == 422
    assert slots_in_use() == 0


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_client_leaving_frees_the_slot(client, fake_copilot, path):
    """The stream's slot must be freed even if the client leaves before its first chunk is read."""
    sent = asyncio.Event()
    fake_copilot.turn = lambda session: sent.set()

    async def scenario() -> None:
        requests = [{"type": "http.request", "body": b'{"prompt": "hi"}', "more_body": False}]

        async def receive() -> dict:
            if requests:
                return requests.pop()
            await sent.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            pass

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"host", b"localhost"), (b"authorization", f"Bearer {API_KEY}".encode()),
                        (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1), "server": ("localhost", 80),
        }
        await asyncio.wait_for(main.app(scope, receive, send), timeout=2)
        await asyncio.gather(*main._closing_sessions)

    client.portal.call(scenario)

    assert slots_in_use() == 0
    assert fake_copilot.sessions_closed == 1


# --- Setting the limits -------------------------------------------------------------------------------------------

LIMITS = [
    ("COPILOT_MAX_PROMPT_CHARS", "_max_prompt_chars", 1_000_000),
    ("COPILOT_MAX_SYSTEM_CHARS", "_max_system_chars", 1_000_000),
    ("COPILOT_MAX_TOOLS", "_max_tools", 128),
    ("COPILOT_MAX_CONCURRENT_SESSIONS", "_max_concurrent_sessions", 20),
]


@pytest.mark.parametrize("variable, attribute, default", LIMITS)
def test_default_limit(monkeypatch, variable, attribute, default):
    monkeypatch.delenv(variable, raising=False)

    with TestClient(main.app):
        assert getattr(main, attribute) == default


@pytest.mark.parametrize("variable, attribute, default", LIMITS)
def test_limit_is_read_from_the_environment_at_startup(monkeypatch, variable, attribute, default):
    monkeypatch.setenv(variable, "7")

    with TestClient(main.app):
        assert getattr(main, attribute) == 7


@pytest.mark.parametrize("variable, attribute, default", LIMITS)
@pytest.mark.parametrize("value", ["", "abc", "0", "-1", "1.5"])
def test_server_refuses_to_start_with_an_invalid_limit(fake_copilot, monkeypatch, variable, attribute, default, value):
    monkeypatch.setenv(variable, value)

    with pytest.raises(RuntimeError, match=variable), TestClient(main.app):
        pass

    assert not fake_copilot.started
