"""GET /models must report a failed model list the same way the routes that depend on that list do."""

import asyncio

import httpx
from conftest import API_KEY

import main

MARKER = "internal-detail-marker"
FAILURE = {"detail": "Could not fetch the Copilot model list"}


def get_models(client) -> httpx.Response:
    """With a time limit, so a stalled list fails the test instead of hanging it."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        headers = {"Authorization": f"Bearer {API_KEY}"}
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost", headers=headers) as http:
            return await asyncio.wait_for(http.get("/models"), timeout=2)

    return client.portal.call(send)


def test_failed_model_list_returns_a_generic_503_and_logs_the_cause(client, fake_copilot, caplog):
    fake_copilot.models_api.error = RuntimeError(MARKER)

    response = get_models(client)

    assert response.status_code == 503
    assert response.json() == FAILURE
    assert MARKER in caplog.text


def test_failure_matches_the_chat_route(client, fake_copilot):
    fake_copilot.models_api.error = RuntimeError(MARKER)

    chat = client.post("/chat", json={"prompt": "hi"})

    assert chat.status_code == get_models(client).status_code == 503
    assert chat.json() == FAILURE


def test_stalled_model_list_returns_503(client, fake_copilot, monkeypatch):
    monkeypatch.setattr(main, "MODEL_LIST_TIMEOUT_SECONDS", 0.01)
    fake_copilot.stalled_calls = {"list_models"}

    response = get_models(client)

    assert response.status_code == 503
    assert response.json() == FAILURE


def test_model_list_works_again_after_a_failure(client, fake_copilot, monkeypatch):
    fake_copilot.models_api.error = RuntimeError(MARKER)
    assert get_models(client).status_code == 503

    async def no_models() -> list:
        return []

    monkeypatch.setattr(fake_copilot, "list_models", no_models)

    response = get_models(client)

    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_client_not_started_still_returns_503(client, monkeypatch):
    monkeypatch.setattr(main, "copilot_client", None)

    response = get_models(client)

    assert response.status_code == 503
    assert response.json() == {"detail": "Copilot client not ready"}
