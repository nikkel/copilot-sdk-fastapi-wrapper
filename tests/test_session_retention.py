"""Single-use sessions must not leave prompts and replies on disk, however the request ended."""

import asyncio

import pytest
from conftest import event
from fastapi.testclient import TestClient
from test_client_disconnect import call_then_disconnect

import main

ALL_ROUTES = ["/chat", "/chat/stream", "/agent/step"]
PROMPT = "private-prompt-text"


def post(client, path, **body):
    return client.post(path, json={"prompt": PROMPT, **body})


@pytest.fixture
def keep_sessions(monkeypatch):
    monkeypatch.setattr(main, "_keep_sessions", True)


# --- Deleted, however the request ended ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_ROUTES)
def test_finished_session_is_closed_then_deleted(client, fake_copilot, path):
    assert post(client, path).status_code == 200

    assert fake_copilot.calls == [("disconnect", "session-0"), ("delete", "session-0")]
    assert fake_copilot.deleted == ["session-0"]


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_session_with_a_session_error_is_deleted(client, fake_copilot, path):
    fake_copilot.turn = lambda session: session.emit(event("session.error", error_type="query", message="failed"))

    post(client, path)

    assert fake_copilot.deleted == ["session-0"]


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_session_with_a_send_failure_is_deleted(client, fake_copilot, path):
    fake_copilot.send_error = RuntimeError("failed")

    post(client, path)

    assert fake_copilot.deleted == ["session-0"]


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_session_with_an_empty_reply_is_deleted(client, fake_copilot, path):
    fake_copilot.turn = lambda session: session.emit(event("session.idle"))

    post(client, path)

    assert fake_copilot.deleted == ["session-0"]


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_timed_out_session_is_deleted(client, fake_copilot, monkeypatch, path):
    monkeypatch.setattr(main, "_request_timeout_seconds", 0.01)
    fake_copilot.turn = lambda session: None

    post(client, path)

    assert fake_copilot.deleted == ["session-0"]


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_disconnected_session_is_deleted(client, fake_copilot, path):
    sent = asyncio.Event()
    fake_copilot.turn = lambda session: sent.set()

    call_then_disconnect(client, path, sent)

    assert fake_copilot.deleted == ["session-0"]


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_session_that_failed_to_close_is_still_deleted(client, fake_copilot, path):
    fake_copilot.disconnect_error = RuntimeError("close failed")

    assert post(client, path).status_code == 200

    assert fake_copilot.deleted == ["session-0"]


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_each_request_deletes_only_its_own_session(client, fake_copilot, path):
    post(client, path)
    post(client, path)

    assert fake_copilot.deleted == ["session-0", "session-1"]


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_session_that_never_opened_has_nothing_to_delete(client, fake_copilot, path):
    fake_copilot.create_session_error = RuntimeError("failed")

    post(client, path)

    assert fake_copilot.calls == []


# --- A failed delete doesn't reach the caller -----------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_ROUTES)
def test_failed_delete_is_logged_without_prompt_text_and_keeps_the_reply(client, fake_copilot, caplog, path):
    fake_copilot.delete_error = RuntimeError(PROMPT)

    response = post(client, path)

    assert response.status_code == 200
    assert "fake reply" in response.text
    assert "event: error" not in response.text
    assert "Could not delete Copilot session session-0" in caplog.text
    assert PROMPT not in caplog.text


@pytest.mark.parametrize("path", ALL_ROUTES)
def test_stalled_delete_is_limited_and_keeps_the_reply(client, fake_copilot, monkeypatch, caplog, path):
    monkeypatch.setattr(main, "SESSION_CLOSE_TIMEOUT_SECONDS", 0.01)
    fake_copilot.stalled_calls = {"delete_session"}

    response = post(client, path)

    assert response.status_code == 200
    assert "Could not delete Copilot session session-0" in caplog.text


def test_delete_has_a_five_second_limit():
    assert main.SESSION_CLOSE_TIMEOUT_SECONDS == 5


# --- Keeping sessions ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_ROUTES)
def test_kept_session_is_closed_and_not_deleted(client, fake_copilot, keep_sessions, path):
    assert post(client, path).status_code == 200

    assert fake_copilot.calls == [("disconnect", "session-0")]
    assert fake_copilot.sessions_closed == 1


def kept_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if "COPILOT_KEEP_SESSIONS" in record.getMessage()]


def test_sessions_are_deleted_by_default(fake_copilot, monkeypatch, caplog):
    monkeypatch.delenv("COPILOT_KEEP_SESSIONS", raising=False)

    with TestClient(main.app):
        assert main._keep_sessions is False

    assert kept_lines(caplog) == []


def test_false_deletes_sessions_and_logs_nothing(fake_copilot, monkeypatch, caplog):
    monkeypatch.setenv("COPILOT_KEEP_SESSIONS", "false")

    with TestClient(main.app):
        assert main._keep_sessions is False

    assert kept_lines(caplog) == []


def test_true_keeps_sessions_and_logs_one_line_at_startup(fake_copilot, monkeypatch, caplog):
    monkeypatch.setenv("COPILOT_KEEP_SESSIONS", "true")

    with TestClient(main.app):
        assert main._keep_sessions is True

    assert len(kept_lines(caplog)) == 1


@pytest.mark.parametrize("value", ["", " ", "True", "FALSE", "1", "0", "yes", "no", " true", "false "])
def test_server_refuses_to_start_on_any_other_value(fake_copilot, monkeypatch, value):
    monkeypatch.setenv("COPILOT_KEEP_SESSIONS", value)

    with pytest.raises(RuntimeError, match="COPILOT_KEEP_SESSIONS"), TestClient(main.app):
        pass

    assert not fake_copilot.started
