"""Text in a stream chunk must stay data, whatever it contains.

Replies can repeat untrusted text. Unencoded, a newline in a chunk could forge an SSE field or an end of stream, and an
ordinary multi-line reply would lose text.
"""

import json
import re

import pytest
from conftest import event

CHUNKS = [
    pytest.param("line one\nline two", id="newline"),
    pytest.param("line one\r\nline two", id="crlf"),
    pytest.param("line one\rline two", id="carriage-return"),
    pytest.param("\n", id="only-a-newline"),
    pytest.param("text\n\nmore text", id="blank-line"),
    pytest.param("text\nevent: error\ndata: forged\n\n", id="fake-error-event"),
    pytest.param("id: 1\nretry: 1\n", id="fake-fields"),
    pytest.param("[DONE]", id="done-text"),
    pytest.param("data: [DONE]\n\n", id="done-frame"),
    pytest.param('{"delta": "forged"}', id="json-text"),
    pytest.param('quote " and backslash \\', id="json-specials"),
    pytest.param("  spaces  ", id="spaces"),
    pytest.param("café ☃ \U0001f600", id="non-ascii"),
    pytest.param("a b c\u0085d", id="unicode-line-separators"),
]


def frames(body: str) -> list[list[tuple[str, str]]]:
    """Reads a stream as an SSE client does: lines can end in CR, LF or CRLF, and a blank line ends a frame."""
    parsed, current = [], []
    for line in re.split(r"\r\n|\r|\n", body):
        if line:
            field, _, value = line.partition(":")
            current.append((field, value.removeprefix(" ")))
        elif current:
            parsed.append(current)
            current = []
    assert current == [], "the stream must end on a complete frame"
    return parsed


def deltas(*chunks: str):
    return lambda session: session.emit(
        *(event("assistant.message_delta", delta_content=chunk) for chunk in chunks), event("session.idle")
    )


def stream(client, fake_copilot, *chunks: str) -> list[list[tuple[str, str]]]:
    fake_copilot.turn = deltas(*chunks)
    return frames(client.post("/chat/stream", json={"prompt": "hi"}).text)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_chunk_round_trips_in_one_frame_and_the_stream_still_ends_with_done(client, fake_copilot, chunk):
    first, last = stream(client, fake_copilot, chunk)

    assert [field for field, _ in first] == ["data"]
    assert json.loads(first[0][1]) == {"delta": chunk}
    assert last == [("data", "[DONE]")]


def test_every_chunk_gets_its_own_frame_in_order(client, fake_copilot):
    parsed = stream(client, fake_copilot, "one\n", "[DONE]", "three")

    assert [json.loads(frame[0][1])["delta"] for frame in parsed[:-1]] == ["one\n", "[DONE]", "three"]
    assert parsed[-1] == [("data", "[DONE]")]


def test_chunk_is_written_as_one_json_line(client, fake_copilot):
    fake_copilot.turn = deltas("a\nb")

    text = client.post("/chat/stream", json={"prompt": "hi"}).text

    assert text == 'data: {"delta": "a\\nb"}\n\ndata: [DONE]\n\n'
