"""A turn that ends with nothing to show must reach the caller as a failure, not as an empty success."""

import json
from types import SimpleNamespace

import pytest
from conftest import event

GENERIC_MESSAGE = "Copilot failed to complete the request"
ERROR_FRAME = f"event: error\ndata: {json.dumps({'message': GENERIC_MESSAGE})}\n\n"
BLOCKING_ROUTES = ["/chat", "/agent/step"]
TOOLS = [{"type": "function", "function": {"name": "search"}}]


def post(client, path, **body):
    return client.post(path, json={"prompt": "hi", **body})


def only_idle(session):
    session.emit(event("session.idle"))


def message_then_idle(content):
    return lambda session: session.emit(event("assistant.message", content=content), event("session.idle"))


def delta_then_idle(delta):
    return lambda session: session.emit(event("assistant.message_delta", delta_content=delta), event("session.idle"))


EMPTY_TURNS = [
    pytest.param(only_idle, id="no-message"),
    pytest.param(message_then_idle(""), id="empty-message"),
    pytest.param(message_then_idle(None), id="null-message"),
    pytest.param(message_then_idle("  \n\t "), id="whitespace-message"),
]


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
@pytest.mark.parametrize("turn", EMPTY_TURNS)
def test_reply_with_no_text_returns_a_generic_502(client, fake_copilot, caplog, path, turn):
    fake_copilot.turn = turn

    response = post(client, path)

    assert response.status_code == 502
    assert response.json() == {"detail": GENERIC_MESSAGE}
    assert "empty reply" in caplog.text
    assert fake_copilot.sessions_closed == 1


@pytest.mark.parametrize("path", BLOCKING_ROUTES)
def test_reply_with_text_still_succeeds(client, fake_copilot, path):
    fake_copilot.turn = message_then_idle(" text ")

    assert post(client, path).status_code == 200


def test_agent_step_with_a_tool_call_and_no_text_still_succeeds(client, fake_copilot):
    def turn(session):
        session.kwargs["tools"][0].handler(SimpleNamespace(arguments={}, tool_name="search"))
        session.emit(event("session.idle"))

    fake_copilot.turn = turn

    response = post(client, "/agent/step", tools=TOOLS)

    assert response.status_code == 200
    assert response.json()["content"] == ""
    assert response.json()["tool_calls"] == [{"name": "search", "arguments": {}}]
    assert response.json()["incomplete"] is False


@pytest.mark.parametrize(
    "turn",
    [pytest.param(only_idle, id="no-chunks"), pytest.param(delta_then_idle(""), id="empty-delta"),
     pytest.param(delta_then_idle(None), id="null-delta")],
)
def test_stream_with_no_chunks_ends_with_the_error_event(client, fake_copilot, caplog, turn):
    fake_copilot.turn = turn

    response = post(client, "/chat/stream")

    assert response.status_code == 200
    assert response.text == ERROR_FRAME
    assert "empty reply" in caplog.text
    assert fake_copilot.sessions_closed == 1


def test_whitespace_only_stream_still_passes(client, fake_copilot):
    """Chunks are sent as they arrive, so whitespace has already reached the caller by the time the turn ends."""
    fake_copilot.turn = delta_then_idle("  \n")

    response = post(client, "/chat/stream")

    assert response.text == 'data: {"delta": "  \\n"}\n\ndata: [DONE]\n\n'
