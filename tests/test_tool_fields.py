"""A malformed request must be refused clearly, before a session opens, rather than failing later in the runtime."""

import pytest

SESSION_ROUTES = ["/chat", "/chat/stream", "/agent/step"]


def step(client, *tools):
    return client.post("/agent/step", json={"prompt": "hi", "tools": list(tools)})


def tool(**function):
    return {"type": "function", "function": {"name": "search", **function}}


@pytest.mark.parametrize(
    "bad_tool",
    [
        pytest.param(tool(parameters="not an object"), id="string-parameters"),
        pytest.param(tool(parameters=1), id="integer-parameters"),
        pytest.param(tool(parameters=["a"]), id="list-parameters"),
        pytest.param(tool(description=1), id="integer-description"),
        pytest.param(tool(description=None), id="null-description"),
        pytest.param(tool(description=["a"]), id="list-description"),
        pytest.param({"type": "other", "function": {"name": "search"}}, id="unknown-type"),
        pytest.param({"type": None, "function": {"name": "search"}}, id="null-type"),
        pytest.param({"type": "function", "function": "search"}, id="string-function"),
        pytest.param({"type": "function"}, id="missing-function"),
    ],
)
def test_malformed_tool_returns_422_and_creates_no_session(client, fake_copilot, bad_tool):
    response = step(client, bad_tool)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:2] == ["body", "tools"]
    assert fake_copilot.sessions == []


def test_duplicate_tool_names_return_422_and_create_no_session(client, fake_copilot):
    response = step(client, tool(), tool(description="the same name again"))

    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"][0]["msg"]
    assert fake_copilot.sessions == []


def test_tool_names_differing_only_by_case_are_distinct(client, fake_copilot):
    response = step(client, tool(), {"type": "function", "function": {"name": "Search"}})

    assert response.status_code == 200
    assert [t.name for t in fake_copilot.sessions[0]["tools"]] == ["search", "Search"]


def test_well_formed_tool_reaches_the_session(client, fake_copilot):
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    response = step(client, tool(description="Searches", parameters=parameters))

    assert response.status_code == 200
    registered = fake_copilot.sessions[0]["tools"][0]
    assert (registered.name, registered.description, registered.parameters) == ("search", "Searches", parameters)


@pytest.mark.parametrize(
    "fields",
    [pytest.param({}, id="omitted"), pytest.param({"parameters": None}, id="null"), pytest.param({"parameters": {}}, id="empty")],
)
def test_tool_without_parameters_gets_an_empty_schema(client, fake_copilot, fields):
    assert step(client, tool(**fields)).status_code == 200

    registered = fake_copilot.sessions[0]["tools"][0]
    assert registered.description == ""
    assert registered.parameters == {"type": "object", "properties": {}}


def test_tool_without_a_type_is_still_a_function(client):
    assert step(client, {"function": {"name": "search"}}).status_code == 200


def test_unknown_function_fields_are_still_ignored(client, fake_copilot):
    """Clients written for other APIs send extras such as strict."""
    assert step(client, tool(strict=True)).status_code == 200
    assert fake_copilot.sessions[0]["tools"][0].name == "search"


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_empty_prompt_returns_422_and_creates_no_session(client, fake_copilot, path):
    response = client.post(path, json={"prompt": ""})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "prompt"]
    assert fake_copilot.sessions == []


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_one_character_prompt_is_accepted(client, path):
    assert client.post(path, json={"prompt": "x"}).status_code == 200
