"""Tests for streaming request-size enforcement."""

import json
from typing import Any

from lung_cancer.middleware import RequestSizeLimitMiddleware


def _run_without_event_loop(coroutine: Any) -> Any:
    """Drive a coroutine made only of immediately completing ASGI test calls."""
    while True:
        try:
            yielded = coroutine.send(None)
        except StopIteration as completed:
            return completed.value
        raise AssertionError(f"Unexpected asynchronous suspension: {yielded!r}")


def _scope(content_length: bytes | None = None) -> dict[str, Any]:
    headers = [] if content_length is None else [(b"content-length", content_length)]
    return {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/classify",
        "headers": headers,
    }


def _response_payload(messages: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    body = next(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(body.decode("utf-8"))


def test_declared_oversized_body_is_rejected_before_app() -> None:
    app_called = False
    messages: list[dict[str, Any]] = []

    async def app(scope, receive, send) -> None:
        nonlocal app_called
        app_called = True

    async def receive() -> dict[str, Any]:
        raise AssertionError("body should not be read after Content-Length rejection")

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    middleware = RequestSizeLimitMiddleware(
        app, max_body_bytes=10, protected_path="/api/v1/classify"
    )
    _run_without_event_loop(middleware(_scope(b"11"), receive, send))

    status, body = _response_payload(messages)
    assert status == 413
    assert "size limit" in body["detail"]
    assert not app_called


def test_streamed_oversized_body_is_rejected() -> None:
    messages: list[dict[str, Any]] = []

    async def app(scope, receive, send) -> None:
        await receive()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"x" * 11, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    middleware = RequestSizeLimitMiddleware(
        app, max_body_bytes=10, protected_path="/api/v1/classify"
    )
    _run_without_event_loop(middleware(_scope(), receive, send))

    status, body = _response_payload(messages)
    assert status == 413
    assert "size limit" in body["detail"]
