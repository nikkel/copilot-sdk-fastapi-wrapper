"""A prompt whose first visible character is / must never reach the runtime, as it may be read as a slash command.

Prompts can carry untrusted text, and the session lists commands such as allow-all and add-dir. Whether session.send
acts on them hasn't been tested, so these prompts are refused before any session opens.
"""

import pytest

ROUTES = ["/chat", "/chat/stream", "/agent/step"]
REFUSED = [
    pytest.param("/help", id="slash"),
    pytest.param("//help", id="double-slash"),
    pytest.param("/", id="only-slash"),
    pytest.param(" /help", id="space"),
    pytest.param("\n\t\r /help", id="mixed-whitespace"),
    pytest.param("\u00a0/help", id="no-break-space"),
    pytest.param("\u3000/help", id="ideographic-space"),
    pytest.param("\u200b/help", id="zero-width-space"),
    pytest.param("\u200d/help", id="zero-width-joiner"),
    pytest.param("\u2060/help", id="word-joiner"),
    pytest.param("\ufeff/help", id="byte-order-mark"),
    pytest.param("\u00ad/help", id="soft-hyphen"),
    pytest.param("\u202e/help", id="right-to-left-override"),
    pytest.param("\U000e0001/help", id="language-tag-outside-the-bmp"),
    pytest.param("\ufeff\u200b\u2060/help", id="several-invisible"),
    pytest.param(" \u200b/help", id="space-then-invisible"),
    pytest.param("\u200b /help", id="invisible-then-space"),
    pytest.param("\ufeff \n\u200b\t\u00a0\u2060/help", id="whitespace-and-invisible-mixed"),
]
ACCEPTED = [
    pytest.param("what does /help do?", id="slash-later"),
    pytest.param("  hello /help", id="whitespace-then-text-then-slash"),
    pytest.param("first line\n/help", id="slash-on-a-later-line"),
    pytest.param("a/b", id="slash-inside-a-word"),
    pytest.param("\u200bhello", id="invisible-then-text"),
    pytest.param("\ufeffhello", id="byte-order-mark-then-text"),
    pytest.param(" \u200b\thello", id="whitespace-and-invisible-then-text"),
    pytest.param("\u200bhello /help", id="invisible-then-text-then-slash"),
    pytest.param("hello\u200b/help", id="invisible-before-a-later-slash"),
]


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("prompt", REFUSED)
def test_prompt_starting_with_a_slash_returns_422_and_creates_no_session(client, fake_copilot, path, prompt):
    response = client.post(path, json={"prompt": prompt})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "prompt"]
    assert fake_copilot.sessions == []
    assert fake_copilot.models_api.calls == 0


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("prompt", ACCEPTED)
def test_prompt_with_a_slash_later_in_the_text_is_accepted(client, fake_copilot, path, prompt):
    response = client.post(path, json={"prompt": prompt})

    assert response.status_code == 200
    assert len(fake_copilot.sessions) == 1
