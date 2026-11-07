"""Only a caller with the API key, using an allowed Host, may reach the API.

Every route but /health can spend the operator's Copilot credits or describe the API. A key stops other local
processes and network callers, and the Host check stops a web page reaching localhost through DNS rebinding.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

import main
from conftest import API_KEY

BODY = {"prompt": "hi"}
SESSION_ROUTES = [
    pytest.param("/chat", id="chat"),
    pytest.param("/chat/stream", id="stream"),
    pytest.param("/agent/step", id="agent-step"),
]
READ_ROUTES = [
    pytest.param("/models", id="models"),
    pytest.param("/docs", id="docs"),
    pytest.param("/redoc", id="redoc"),
    pytest.param("/openapi.json", id="openapi"),
]
BAD_AUTHORIZATION = [
    pytest.param("Bearer wrong-key", id="wrong-key"),
    pytest.param(f"Bearer {API_KEY}x", id="key-with-suffix"),
    pytest.param(f"Basic {API_KEY}", id="wrong-scheme"),
    pytest.param(API_KEY, id="no-scheme"),
    pytest.param("Bearer ", id="empty-key"),
    pytest.param("Bearer café".encode("latin-1"), id="non-ascii"),
]
AUTHORIZED = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture(autouse=True)
def serializable_models(fake_copilot, monkeypatch):
    """GET /models serialises what the SDK returns. These tests only care whether the route is reached."""

    async def no_models() -> list:
        return []

    monkeypatch.setattr(fake_copilot, "list_models", no_models)


def assert_refused(response) -> None:
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "Missing or invalid API key"}


# --- API key ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_session_route_without_key_returns_401_and_creates_no_session(anonymous, fake_copilot, path):
    assert_refused(anonymous.post(path, json=BODY))
    assert fake_copilot.sessions == []


@pytest.mark.parametrize("path", SESSION_ROUTES)
@pytest.mark.parametrize("authorization", BAD_AUTHORIZATION)
def test_session_route_with_bad_key_returns_401_and_creates_no_session(anonymous, fake_copilot, path, authorization):
    assert_refused(anonymous.post(path, json=BODY, headers={"Authorization": authorization}))
    assert fake_copilot.sessions == []


@pytest.mark.parametrize("path", READ_ROUTES)
def test_read_route_without_key_returns_401(anonymous, path):
    assert_refused(anonymous.get(path))


def test_unknown_path_without_key_returns_401(anonymous):
    assert_refused(anonymous.get("/not-a-route"))


@pytest.mark.parametrize("path", ["/health/", "//health", "/health/x"])
def test_only_the_exact_health_path_is_open(anonymous, path):
    assert_refused(anonymous.get(path, follow_redirects=False))


def test_websocket_without_key_is_refused(anonymous):
    with pytest.raises(WebSocketDenialResponse) as denied:
        with anonymous.websocket_connect("ws://localhost/not-a-route"):
            pass
    assert denied.value.status_code == 401


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_session_route_with_key_succeeds(client, fake_copilot, path):
    response = client.post(path, json=BODY)

    assert response.status_code == 200
    assert len(fake_copilot.sessions) == 1


@pytest.mark.parametrize("path", READ_ROUTES)
def test_read_route_with_key_succeeds(client, path):
    assert client.get(path).status_code == 200


def test_scheme_is_case_insensitive(anonymous):
    assert anonymous.get("/models", headers={"Authorization": f"bearer {API_KEY}"}).status_code == 200


def test_health_needs_no_key(anonymous):
    assert anonymous.get("/health").json() == {"status": "ok"}


def test_key_is_compared_in_constant_time(anonymous, monkeypatch):
    compared = []
    real_compare = main.secrets.compare_digest
    monkeypatch.setattr(main.secrets, "compare_digest", lambda a, b: compared.append((a, b)) or real_compare(a, b))

    anonymous.get("/models", headers={"Authorization": "Bearer wrong-key"})

    assert compared == [(b"wrong-key", API_KEY.encode())]


def test_requests_are_refused_if_startup_did_not_load_a_key(fake_copilot):
    """With uvicorn's --lifespan off the startup check never runs, so a missing key must still deny."""
    unstarted = TestClient(main.app, base_url="http://localhost", headers=AUTHORIZED)

    assert_refused(unstarted.post("/chat", json=BODY))


def test_copilot_runtime_does_not_inherit_the_key(fake_copilot, monkeypatch):
    options = {}
    monkeypatch.setattr(main, "CopilotClient", lambda **kwargs: options.update(kwargs) or fake_copilot)

    with TestClient(main.app):
        pass

    assert "WRAPPER_API_KEY" not in options["env"]
    assert options["env"]["COPILOT_WRAPPER_HOME"] == os.environ["COPILOT_WRAPPER_HOME"]


# --- Refusing to start --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_server_refuses_to_start_without_a_key(fake_copilot, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("WRAPPER_API_KEY")
    else:
        monkeypatch.setenv("WRAPPER_API_KEY", value)

    with pytest.raises(RuntimeError, match="WRAPPER_API_KEY"):
        with TestClient(main.app):
            pass

    assert not fake_copilot.started


# --- Host allowlist -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_foreign_host_with_valid_key_returns_400_and_creates_no_session(client, fake_copilot, path):
    response = client.post(path, json=BODY, headers={"Host": "attacker.example"})

    assert response.status_code == 400
    assert fake_copilot.sessions == []


@pytest.mark.parametrize("path", ["/health", "/models", "/docs", "/openapi.json", "/not-a-route"])
def test_foreign_host_is_refused_on_every_path(client, path):
    assert client.get(path, headers={"Host": "attacker.example"}).status_code == 400


def test_foreign_host_without_key_gets_400_not_401(anonymous):
    assert anonymous.post("/chat", json=BODY, headers={"Host": "attacker.example"}).status_code == 400


@pytest.mark.parametrize(
    "host", ["localhost", "localhost:8000", "127.0.0.1", "127.0.0.1:8000", "LOCALHOST"]
)
def test_local_host_is_accepted(client, host):
    assert client.post("/chat", json=BODY, headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "host", ["localhost.attacker.example", "127.0.0.1.attacker.example", "evil-localhost", "testserver"]
)
def test_lookalike_host_is_refused(client, host):
    assert client.post("/chat", json=BODY, headers={"Host": host}).status_code == 400


def test_default_allowed_hosts(monkeypatch):
    monkeypatch.delenv("WRAPPER_ALLOWED_HOSTS", raising=False)
    assert main._allowed_hosts() == ["localhost", "127.0.0.1"]


@pytest.mark.parametrize("value", ["", "  ", " , "])
def test_blank_allowed_hosts_keep_the_default(monkeypatch, value):
    monkeypatch.setenv("WRAPPER_ALLOWED_HOSTS", value)
    assert main._allowed_hosts() == ["localhost", "127.0.0.1"]


def test_allowed_hosts_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("WRAPPER_ALLOWED_HOSTS", "api.internal, host.docker.internal,")
    assert main._allowed_hosts() == ["api.internal", "host.docker.internal"]


@pytest.mark.parametrize("value", ["*", "localhost,*", "*.example.com"])
def test_wildcard_allowed_hosts_are_refused(monkeypatch, value):
    monkeypatch.setenv("WRAPPER_ALLOWED_HOSTS", value)
    with pytest.raises(RuntimeError, match="WRAPPER_ALLOWED_HOSTS"):
        main._allowed_hosts()


def test_configured_allowed_hosts_replace_the_default_on_the_app():
    """The middleware is built when main is imported, so this needs a fresh process."""
    script = (
        "from fastapi.testclient import TestClient\n"
        "import main\n"
        "client = TestClient(main.app)\n"
        "for host in ('api.internal', 'localhost'):\n"
        "    print(client.get('/health', headers={'Host': host}).status_code)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(main.__file__).parent,
        env={**os.environ, "WRAPPER_ALLOWED_HOSTS": "api.internal"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.split() == ["200", "400"]
