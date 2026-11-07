"""A fake Copilot SDK client, so the tests never start the runtime, reach GitHub or spend credits."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main

API_KEY = "test-api-key"


def _event(event_type: str, **data) -> SimpleNamespace:
    return SimpleNamespace(type=SimpleNamespace(value=event_type), data=SimpleNamespace(**data))


class FakeSession:
    def __init__(self, streaming: bool) -> None:
        self.streaming = streaming
        self.handlers = []

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def on(self, handler) -> None:
        self.handlers.append(handler)

    async def send(self, prompt: str) -> None:
        if self.streaming:
            reply = _event("assistant.message_delta", delta_content="fake reply")
        else:
            reply = _event("assistant.message", content="fake reply")
        for event in (reply, _event("session.idle")):
            for handler in self.handlers:
                handler(event)


class FakeModelsApi:
    """Stands in for client.rpc.models: every call is a fresh fetch from the runtime."""

    def __init__(self) -> None:
        self.ids = ["auto", "gpt-6-luna", "gpt-5-mini"]
        self.error: Exception | None = None
        self.calls = 0

    async def list(self, params, *, timeout: float | None = None) -> SimpleNamespace:
        self.calls += 1
        await asyncio.sleep(0)  # let concurrent requests interleave, as a real round trip would
        if self.error is not None:
            raise self.error
        return SimpleNamespace(models=[SimpleNamespace(id=model_id) for model_id in self.ids])


class FakeCopilotClient:
    def __init__(self) -> None:
        self.models_api = FakeModelsApi()
        self.rpc = SimpleNamespace(models=self.models_api)
        self.sessions: list[dict] = []
        self._models_cache = None
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        return None

    async def list_models(self) -> list[SimpleNamespace]:
        """Like the real SDK: fetches once, then returns that result until the client disconnects."""
        if self._models_cache is None:
            self._models_cache = (await self.models_api.list(None)).models
        return list(self._models_cache)

    async def create_session(self, **kwargs) -> FakeSession:
        self.sessions.append(kwargs)
        return FakeSession(streaming=kwargs.get("streaming", False))


@pytest.fixture(autouse=True)
def fake_copilot(monkeypatch, tmp_path) -> FakeCopilotClient:
    """Applies to every test, so none of them can construct the real client."""
    fake = FakeCopilotClient()
    monkeypatch.setattr(main, "CopilotClient", lambda **kwargs: fake)
    monkeypatch.setattr(main, "copilot_client", None)
    monkeypatch.setattr(main, "_api_key", None, raising=False)
    monkeypatch.setenv("COPILOT_WRAPPER_HOME", str(tmp_path))
    monkeypatch.setenv("WRAPPER_API_KEY", API_KEY)
    return fake


@pytest.fixture
def client(fake_copilot):
    """An authorised local caller, with the app's lifespan running so each test starts from a fresh process state."""
    with TestClient(
        main.app, base_url="http://localhost", headers={"Authorization": f"Bearer {API_KEY}"}
    ) as test_client:
        yield test_client


@pytest.fixture
def anonymous(fake_copilot):
    """A local caller that sends no API key, with the app's lifespan running."""
    with TestClient(main.app, base_url="http://localhost") as test_client:
        yield test_client


@pytest.fixture
def clock(monkeypatch) -> SimpleNamespace:
    """A controllable clock for the model cache. Only main's reference is replaced, not the event loop's."""
    state = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(main, "monotonic", lambda: state.now, raising=False)
    return state
