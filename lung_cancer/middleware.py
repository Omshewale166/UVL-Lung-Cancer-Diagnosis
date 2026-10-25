"""ASGI request-body limits for potentially large image uploads."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _RequestBodyTooLarge(Exception):
    """Raised when a streamed request exceeds its configured byte limit."""


class RequestSizeLimitMiddleware:
    """Reject oversized classification requests before parsing file uploads."""

    def __init__(
        self, app: ASGIApp, max_body_bytes: int, protected_path: str
    ) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.protected_path = protected_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Limit both declared and streamed request bodies on the upload route."""
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self.protected_path
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                await JSONResponse(
                    {"detail": "Invalid Content-Length header"}, status_code=400
                )(scope, receive, send)
                return
            if content_length < 0:
                await JSONResponse(
                    {"detail": "Invalid Content-Length header"}, status_code=400
                )(scope, receive, send)
                return
            if content_length > self.max_body_bytes:
                await self._too_large(scope, receive, send)
                return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if isinstance(body, bytes):
                    received_bytes += len(body)
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def track_response(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, track_response)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await self._too_large(scope, receive, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Return the standard upload-size error response."""
        response = JSONResponse(
            {"detail": "Request body exceeds configured size limit"},
            status_code=413,
        )
        await response(scope, receive, send)
