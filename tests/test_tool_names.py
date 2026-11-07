"""A caller's tool name must only ever name a tool.

/agent/step copies tool names into the session's tool allowlist, where the runtime reads entries such as a
source-qualified wildcard as filter patterns. A name written as a pattern could allow Copilot's built-in tools.
"""

import pytest

TOOL_PATTERNS = ["builtin:*", "builtin:bash", "mcp:*", "custom:*", "*"]
OTHER_BAD_NAMES = ["", "has space", "search\n", "x" * 65, 123, None, ["search"]]


def step(client, *functions):
    tools = [{"type": "function", "function": function} for function in functions]
    return client.post("/agent/step", json={"prompt": "hi", "tools": tools})


@pytest.mark.parametrize("name", TOOL_PATTERNS + OTHER_BAD_NAMES)
def test_invalid_tool_name_returns_422_and_never_creates_a_session(client, fake_copilot, name):
    response = step(client, {"name": name})

    assert response.status_code == 422
    assert fake_copilot.sessions == []


def test_tool_without_a_name_returns_422_and_never_creates_a_session(client, fake_copilot):
    response = step(client, {"description": "no name"})

    assert response.status_code == 422
    assert fake_copilot.sessions == []


def test_one_invalid_tool_rejects_the_whole_request(client, fake_copilot):
    response = step(client, {"name": "search"}, {"name": "builtin:*"})

    assert response.status_code == 422
    assert fake_copilot.sessions == []


@pytest.mark.parametrize("name", ["search", "web_fetch", "get-page", "Tool2", "x" * 64])
def test_valid_tool_name_is_the_only_allowlist_entry(client, fake_copilot, name):
    response = step(client, {"name": name})

    assert response.status_code == 200
    assert [session["available_tools"] for session in fake_copilot.sessions] == [[name]]
    assert [tool.name for tool in fake_copilot.sessions[0]["tools"]] == [name]


def test_no_tools_gives_an_empty_allowlist(client, fake_copilot):
    response = client.post("/agent/step", json={"prompt": "hi"})

    assert response.status_code == 200
    assert fake_copilot.sessions[0]["available_tools"] == []
