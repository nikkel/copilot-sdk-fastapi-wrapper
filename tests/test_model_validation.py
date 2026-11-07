"""Every route that opens a session must refuse a model the account doesn't list.

Copilot silently answers an unknown model with a different one, which can cost far more than the caller chose.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from conftest import API_KEY
from copilot.generated.rpc import ServerRpc
from fastapi.testclient import TestClient

import main

TOOLS = [{"type": "function", "function": {"name": "search"}}]
ROUTES = [
    pytest.param("/chat", {}, id="chat"),
    pytest.param("/chat/stream", {}, id="stream"),
    pytest.param("/agent/step", {"tools": TOOLS}, id="agent-step"),
]
all_routes = pytest.mark.parametrize("path, extra", ROUTES)


def post(client, path, extra, model):
    return client.post(path, json={"prompt": "hi", "model": model, **extra})


# --- Unknown and known models -------------------------------------------------------------------------------------

@all_routes
def test_unknown_model_returns_422_and_never_creates_a_session(client, fake_copilot, path, extra):
    response = post(client, path, extra, "not-a-real-model")

    assert response.status_code == 422
    assert fake_copilot.sessions == []
    valid_models = response.json()["detail"]["valid_models"]
    assert set(valid_models) == {"auto", "gpt-6-luna", "gpt-5-mini"}


@all_routes
def test_listed_model_reaches_the_session_and_gets_a_reply(client, fake_copilot, path, extra):
    response = post(client, path, extra, "gpt-5-mini")

    assert response.status_code == 200
    assert "fake reply" in response.text
    assert [session["model"] for session in fake_copilot.sessions] == ["gpt-5-mini"]


@pytest.mark.parametrize("model", ["GPT-5-MINI", "gpt-5-mini ", ""])
def test_near_misses_are_not_accepted(client, fake_copilot, model):
    response = post(client, "/chat", {}, model)

    assert response.status_code == 422
    assert fake_copilot.sessions == []


def test_valid_id_list_is_short_and_puts_a_close_match_first(client, fake_copilot):
    fake_copilot.models_api.ids = [f"aaa-model-{n}" for n in range(50)] + ["zzz-target"]

    response = post(client, "/chat", {}, "zzz-targte")

    valid_models = response.json()["detail"]["valid_models"]
    assert response.status_code == 422
    assert len(valid_models) <= 10
    assert valid_models[0] == "zzz-target"


# --- "auto" -------------------------------------------------------------------------------------------------------

@all_routes
def test_auto_is_valid_when_the_account_lists_it(client, fake_copilot, path, extra):
    response = post(client, path, extra, "auto")

    assert response.status_code == 200
    assert [session["model"] for session in fake_copilot.sessions] == ["auto"]


@all_routes
def test_auto_is_rejected_when_the_account_does_not_list_it(client, fake_copilot, path, extra):
    fake_copilot.models_api.ids = ["gpt-6-luna"]

    response = post(client, path, extra, "auto")

    assert response.status_code == 422
    assert fake_copilot.sessions == []


# --- The model list can't be fetched ------------------------------------------------------------------------------

@all_routes
@pytest.mark.parametrize("error", [RuntimeError("not connected"), TimeoutError()], ids=["runtime-error", "timeout"])
def test_fetch_failure_returns_503_and_never_creates_a_session(client, fake_copilot, path, extra, error):
    fake_copilot.models_api.error = error

    response = post(client, path, extra, "gpt-6-luna")

    assert response.status_code == 503
    assert fake_copilot.sessions == []


@all_routes
def test_empty_model_list_returns_503_and_never_creates_a_session(client, fake_copilot, path, extra):
    fake_copilot.models_api.ids = []

    response = post(client, path, extra, "gpt-6-luna")

    assert response.status_code == 503
    assert fake_copilot.sessions == []


@all_routes
def test_client_not_started_returns_503(monkeypatch, path, extra):
    monkeypatch.setattr(main, "_api_key", API_KEY.encode())
    unstarted = TestClient(main.app, base_url="http://localhost", headers={"Authorization": f"Bearer {API_KEY}"})

    response = post(unstarted, path, extra, "gpt-6-luna")

    assert response.status_code == 503


def test_a_failed_fetch_is_not_cached(client, fake_copilot):
    fake_copilot.models_api.error = RuntimeError("runtime restarting")
    assert post(client, "/chat", {}, "gpt-6-luna").status_code == 503

    fake_copilot.models_api.error = None

    assert post(client, "/chat", {}, "gpt-6-luna").status_code == 200


def test_expired_list_is_not_used_when_the_refresh_fails(client, fake_copilot, clock):
    assert post(client, "/chat", {}, "gpt-6-luna").status_code == 200
    fake_copilot.sessions.clear()

    clock.now += 300
    fake_copilot.models_api.error = RuntimeError("runtime restarting")

    assert post(client, "/chat", {}, "gpt-6-luna").status_code == 503
    assert fake_copilot.sessions == []


# --- The five-minute cache ----------------------------------------------------------------------------------------

def test_list_is_fetched_once_within_five_minutes_across_all_routes(client, fake_copilot, clock):
    assert post(client, "/chat", {}, "gpt-6-luna").status_code == 200
    clock.now += 299.9
    assert post(client, "/chat/stream", {}, "gpt-6-luna").status_code == 200
    assert post(client, "/agent/step", {"tools": TOOLS}, "not-a-real-model").status_code == 422

    assert fake_copilot.models_api.calls == 1


def test_model_added_upstream_is_rejected_until_the_cache_expires(client, fake_copilot, clock):
    assert post(client, "/chat", {}, "gpt-6-luna").status_code == 200
    fake_copilot.models_api.ids = ["gpt-6-luna", "brand-new-model"]

    clock.now += 299.9
    assert post(client, "/chat", {}, "brand-new-model").status_code == 422

    clock.now += 0.1
    assert post(client, "/chat", {}, "brand-new-model").status_code == 200
    assert fake_copilot.models_api.calls == 2


def test_model_removed_upstream_is_rejected_after_the_cache_expires(client, fake_copilot, clock):
    assert post(client, "/chat", {}, "gpt-5-mini").status_code == 200
    fake_copilot.sessions.clear()
    fake_copilot.models_api.ids = ["gpt-6-luna"]

    clock.now += 300

    assert post(client, "/chat", {}, "gpt-5-mini").status_code == 422
    assert fake_copilot.sessions == []


def test_concurrent_requests_on_a_cold_cache_fetch_once(client, fake_copilot):
    async def send_together() -> list[int]:
        transport = httpx.ASGITransport(app=main.app)
        headers = {"Authorization": f"Bearer {API_KEY}"}
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost", headers=headers) as http:
            responses = await asyncio.gather(
                *(http.post("/chat", json={"prompt": "hi", "model": "gpt-6-luna"}) for _ in range(5))
            )
        return [response.status_code for response in responses]

    assert client.portal.call(send_together) == [200] * 5
    assert fake_copilot.models_api.calls == 1


# --- The fake's model API matches the real SDK ----------------------------------------------------------------------

def test_fetch_uses_the_real_sdk_rpc_with_a_timeout(client, fake_copilot, monkeypatch):
    """Drives the SDK's real RPC and parsing with only the transport stubbed, so the fake can't drift unnoticed."""
    requests = []

    async def request(method, params, **kwargs):
        requests.append((method, kwargs))
        return {"models": [{"id": "from-real-parser", "name": "Real parser", "capabilities": {}}]}

    fake_copilot.rpc = ServerRpc(SimpleNamespace(request=request))

    assert post(client, "/chat", {}, "from-real-parser").status_code == 200
    assert post(client, "/chat", {}, "not-a-real-model").status_code == 422
    assert len(requests) == 1
    method, kwargs = requests[0]
    assert method == "models.list"
    assert kwargs["timeout"] > 0
