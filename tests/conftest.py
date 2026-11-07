"""A fake Copilot SDK client, so the tests never start the runtime, reach GitHub or spend credits."""

import asyncio
from types import SimpleNamespace
from typing import Self

import pytest
from fastapi.testclient import TestClient

import main

API_KEY = "test-api-key"


def event(event_type: str, **data) -> SimpleNamespace:
    return SimpleNamespace(type=SimpleNamespace(value=event_type), data=SimpleNamespace(**data))


class FakeSession:
    def __init__(self, client: "FakeCopilotClient", kwargs: dict) -> None:
        self.client = client
        self.kwargs = kwargs
        self.session_id = f"session-{len(client.sessions) - 1}"
        self.streaming = kwargs.get("streaming", False)
        self.handlers = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.disconnect()

    async def disconnect(self) -> None:
        await asyncio.sleep(0)  # a real round trip, so a cancellation still being delivered would interrupt this
        self.client.calls.append(("disconnect", self.session_id))
        await self.client.maybe_stall("disconnect")
        if self.client.disconnect_error is not None:
            raise self.client.disconnect_error
        self.client.sessions_closed += 1

    async def abort(self) -> None:
        await asyncio.sleep(0)
        self.client.turns_aborted += 1
        await self.client.maybe_stall("abort")

    def on(self, handler) -> None:
        self.handlers.append(handler)

    async def send(self, prompt: str) -> None:
        await self.client.maybe_stall("send")
        if self.client.send_error is not None:
            raise self.client.send_error
        if self.client.turn is not None:
            self.client.turn(self)
            return
        if self.streaming:
            reply = event("assistant.message_delta", delta_content="fake reply")
        else:
            reply = event("assistant.message", content="fake reply")
        self.emit(reply, event("session.idle"))

    def emit(self, *events: SimpleNamespace) -> None:
        for event in events:
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
        self.sessions_closed = 0
        self.calls: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.delete_error: Exception | None = None
        self.turns_aborted = 0
        self.stalled_calls: set[str] = set()  # names of SDK calls that never return
        self.disconnect_error: Exception | None = None
        self.turn = None  # replaces the normal reply: called with the session when a prompt is sent
        self.send_error: Exception | None = None
        self.create_session_error: Exception | None = None

    async def maybe_stall(self, call: str) -> None:
        if call in self.stalled_calls:
            await asyncio.Event().wait()

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        return None

    async def list_models(self) -> list[SimpleNamespace]:
        """Like the real SDK: fetches once, then returns that result until the client disconnects."""
        await self.maybe_stall("list_models")
        if self._models_cache is None:
            self._models_cache = (await self.models_api.list(None)).models
        return list(self._models_cache)

    async def delete_session(self, session_id: str) -> None:
        await asyncio.sleep(0)
        self.calls.append(("delete", session_id))
        await self.maybe_stall("delete_session")
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(session_id)

    async def create_session(self, **kwargs) -> FakeSession:
        await self.maybe_stall("create_session")
        if self.create_session_error is not None:
            raise self.create_session_error
        self.sessions.append(kwargs)
        return FakeSession(self, kwargs)


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
