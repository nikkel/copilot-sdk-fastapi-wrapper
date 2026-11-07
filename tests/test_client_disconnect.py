"""A turn nobody is waiting for must stop, so an abandoned request doesn't keep spending or holding a session."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import API_KEY, event

import main

ALL_ROUTES = ["/chat", "/agent/step", "/chat/stream"]
BLOCKING_ROUTES = ["/chat", "/agent/step"]
BODY = json.dumps({"prompt": "hi"}).encode()
ROOT = Path(main.__file__).parent


def scope_for(path: str, body: bytes = BODY) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},  # what uvicorn sends
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"localhost"),
            (b"authorization", f"Bearer {API_KEY}".encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("localhost", 80),
    }


def call_then_disconnect(client, path: str, leave_when: asyncio.Event | None = None, body: bytes = BODY) -> list[dict]:
    """Runs a request through the app, as uvicorn does. The client leaves when the event is set, or straight away."""

    async def run() -> list[dict]:
        sent: list[dict] = []
        requests = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive() -> dict:
            if requests:
                return requests.pop()
            if leave_when is not None:
                await leave_when.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            sent.append(message)

        await asyncio.wait_for(main.app(scope_for(path, body), receive, send), timeout=2)
        await asyncio.gather(*main._closing_sessions)  # a close can outlive the cancelled request that started it
        return sent

    return client.portal.call(run)


def leave_once_the_prompt_is_sent(fake_copilot, *events) -> asyncio.Event:
    sent = asyncio.Event()

    def turn(session):
        session.emit(*events)
        sent.set()

    fake_copilot.turn = turn
    return sent


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_client_leaving_during_a_stalled_turn_aborts_and_closes_the_session(client, fake_copilot, path):
    sent = leave_once_the_prompt_is_sent(fake_copilot)

    messages = call_then_disconnect(client, path, sent)

    assert not any(message.get("status") == 200 for message in messages if path != "/chat/stream")
    assert fake_copilot.turns_aborted == 1
    assert fake_copilot.sessions_closed == 1


def test_client_leaving_a_stream_after_a_chunk_aborts_and_closes_the_session(client, fake_copilot):
    sent = leave_once_the_prompt_is_sent(fake_copilot, event("assistant.message_delta", delta_content="part"))

    call_then_disconnect(client, "/chat/stream", sent)

    assert fake_copilot.turns_aborted == 1
    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_client_leaving_during_a_stalled_send_closes_the_session(client, fake_copilot, path):
    fake_copilot.stalled_calls = {"send"}

    call_then_disconnect(client, path)

    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_client_leaving_during_a_stalled_create_session_ends_the_request(client, fake_copilot, path):
    fake_copilot.stalled_calls = {"create_session"}

    call_then_disconnect(client, path)

    assert fake_copilot.sessions_closed == 0


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_stalled_abort_after_the_client_leaves_still_closes_the_session(client, fake_copilot, monkeypatch, path):
    monkeypatch.setattr(main, "SESSION_CLOSE_TIMEOUT_SECONDS", 0.05)
    sent = leave_once_the_prompt_is_sent(fake_copilot)
    fake_copilot.stalled_calls = {"abort"}

    call_then_disconnect(client, path, sent)

    assert fake_copilot.sessions_closed == 1


def test_client_leaving_after_a_tool_call_closes_the_session(client, fake_copilot):
    sent = asyncio.Event()

    def turn(session):
        session.kwargs["tools"][0].handler(SimpleNamespace(arguments={}, tool_name="search"))
        sent.set()

    fake_copilot.turn = turn
    body = json.dumps({"prompt": "hi", "tools": [{"type": "function", "function": {"name": "search"}}]}).encode()

    call_then_disconnect(client, "/agent/step", sent, body)

    assert fake_copilot.turns_aborted == 1
    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_a_client_that_stays_gets_its_reply_and_the_turn_is_not_aborted(client, fake_copilot, path):
    assert client.post(path, json={"prompt": "hi"}).status_code == 200

    assert fake_copilot.turns_aborted == 0
    assert fake_copilot.sessions_closed == 1
    assert main._closing_sessions == set()


# --- SIGTERM with a stalled turn: a real server process, using the fake client ------------------------------------

SERVER = """
import sys
sys.path[:0] = [".", "tests"]
import main, uvicorn
from conftest import FakeCopilotClient

fake = FakeCopilotClient()
main.CopilotClient = lambda **kwargs: fake
fake.turn = lambda session: print("TURN STARTED", flush=True)

real_create_session = fake.create_session
async def create_session(**kwargs):
    session = await real_create_session(**kwargs)
    disconnect = session.disconnect
    abort = session.abort
    async def noisy_abort():
        await abort()
        print("TURN ABORTED", flush=True)
    async def noisy_disconnect():
        await disconnect()
        print("SESSION CLOSED", flush=True)
    session.abort, session.disconnect = noisy_abort, noisy_disconnect
    return session
fake.create_session = create_session

uvicorn.run(main.app, host="127.0.0.1", port=int(sys.argv[1]), timeout_graceful_shutdown=None if sys.argv[2] == "None" else int(sys.argv[2]))
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def sigterm_during_a_stalled_turn(tmp_path, graceful_shutdown: str, request_timeout: str) -> list[str]:
    """Returns the server's output. Reading it blocks until each step happens, so nothing here needs a sleep."""
    port = free_port()
    env = {
        **os.environ,
        "WRAPPER_API_KEY": API_KEY,
        "COPILOT_WRAPPER_HOME": str(tmp_path),
        "COPILOT_REQUEST_TIMEOUT": request_timeout,
    }
    server = subprocess.Popen(
        [sys.executable, "-c", SERVER, str(port), graceful_shutdown],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Fails the test rather than hanging the run if the server never exits.
    watchdog = threading.Timer(20, server.kill)
    watchdog.start()
    output: list[str] = []
    try:
        for line in server.stdout:
            output.append(line.strip())
            if "Uvicorn running" in line:
                break
        with socket.create_connection(("127.0.0.1", port)) as connection:
            connection.sendall(
                b"POST /chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                + f"Authorization: Bearer {API_KEY}\r\nContent-Length: {len(BODY)}\r\n\r\n".encode()
                + BODY
            )
            for line in server.stdout:
                output.append(line.strip())
                if "TURN STARTED" in line:
                    break
            server.send_signal(signal.SIGTERM)
            output.extend(line.strip() for line in server.stdout)
            server.wait()
    finally:
        was_killed = not watchdog.is_alive()
        watchdog.cancel()
        server.kill()
    assert not was_killed, f"the server didn't exit after SIGTERM: {output}"
    return output


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGTERM")
def test_sigterm_with_a_graceful_shutdown_limit_ends_a_stalled_turn(tmp_path):
    output = sigterm_during_a_stalled_turn(tmp_path, graceful_shutdown="1", request_timeout="120")

    assert "TURN ABORTED" in output
    assert "SESSION CLOSED" in output


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGTERM")
def test_sigterm_with_no_graceful_shutdown_limit_exits_when_the_turn_times_out(tmp_path):
    output = sigterm_during_a_stalled_turn(tmp_path, graceful_shutdown="None", request_timeout="1")

    assert "TURN ABORTED" in output
    assert "SESSION CLOSED" in output
