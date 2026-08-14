# 공식 가중치 HTTP 경계의 크기와 리다이렉트 제한을 검증합니다.
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from k3x_converter.format import K3XError
from k3x_converter.official_transport import UrllibTransport


class _Handler(BaseHTTPRequestHandler):
    requests = 0

    def do_GET(self) -> None:  # noqa: N802
        type(self).requests += 1
        if self.path == "/ok":
            self.send_response(200)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"hello")
        elif self.path == "/large":
            self.send_response(200)
            self.send_header("Content-Length", "6")
            self.end_headers()
            self.wfile.write(b"123456")
        elif self.path == "/redirect-good":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.end_headers()
        elif self.path == "/redirect-bad":
            self.send_response(302)
            self.send_header("Location", "https://example.com/escape")
            self.end_headers()
        elif self.path == "/redirect-loop":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop")
            self.end_headers()
        else:
            self.send_response(201)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def local_server() -> str:
    _Handler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_transport_returns_bounded_body_and_real_counters(local_server: str) -> None:
    transport = UrllibTransport(allowed_hosts=frozenset({"127.0.0.1"}))

    response = transport.get(
        f"{local_server}/ok", headers={}, max_bytes=5, timeout_seconds=2.0
    )

    assert response.status == 200
    assert response.body == b"hello"
    assert response.final_url == f"{local_server}/ok"
    assert transport.stats.requests == 1
    assert transport.stats.response_bytes == 5
    assert transport.stats.maximum_response_bytes == 5


def test_transport_reuses_digest_validated_persistent_cache(
    local_server: str, tmp_path: Path
) -> None:
    first = UrllibTransport(
        allowed_hosts=frozenset({"127.0.0.1"}), cache_directory=tmp_path
    )
    assert first.get(
        f"{local_server}/ok", headers={}, max_bytes=5, timeout_seconds=2.0
    ).body == b"hello"
    assert _Handler.requests == 1

    second = UrllibTransport(
        allowed_hosts=frozenset({"127.0.0.1"}), cache_directory=tmp_path
    )
    assert second.get(
        f"{local_server}/ok", headers={}, max_bytes=5, timeout_seconds=2.0
    ).body == b"hello"
    assert _Handler.requests == 1
    assert second.stats.requests == 0

    next(tmp_path.glob("*.body")).write_bytes(b"jello")
    third = UrllibTransport(
        allowed_hosts=frozenset({"127.0.0.1"}), cache_directory=tmp_path
    )
    assert third.get(
        f"{local_server}/ok", headers={}, max_bytes=5, timeout_seconds=2.0
    ).body == b"hello"
    assert _Handler.requests == 2


def test_transport_rejects_body_one_byte_over_limit(local_server: str) -> None:
    transport = UrllibTransport(allowed_hosts=frozenset({"127.0.0.1"}))

    with pytest.raises(K3XError, match="OFFICIAL_BODY_LIMIT"):
        transport.get(
            f"{local_server}/large", headers={}, max_bytes=5, timeout_seconds=2.0
        )


def test_transport_validates_every_redirect_host(local_server: str) -> None:
    transport = UrllibTransport(allowed_hosts=frozenset({"127.0.0.1"}))
    assert transport.get(
        f"{local_server}/redirect-good",
        headers={},
        max_bytes=5,
        timeout_seconds=2.0,
    ).body == b"hello"

    with pytest.raises(K3XError, match="UNTRUSTED_OFFICIAL_HOST"):
        transport.get(
            f"{local_server}/redirect-bad",
            headers={},
            max_bytes=5,
            timeout_seconds=2.0,
        )


def test_transport_rejects_redirect_limit_and_status_drift(local_server: str) -> None:
    transport = UrllibTransport(
        max_redirects=2, allowed_hosts=frozenset({"127.0.0.1"})
    )
    with pytest.raises(K3XError, match="OFFICIAL_REDIRECT_LIMIT"):
        transport.get(
            f"{local_server}/redirect-loop",
            headers={},
            max_bytes=1,
            timeout_seconds=2.0,
        )
    with pytest.raises(K3XError, match="OFFICIAL_HTTP_STATUS"):
        transport.get(
            f"{local_server}/wrong",
            headers={},
            max_bytes=0,
            timeout_seconds=2.0,
        )


def test_transport_rejects_untrusted_initial_host_before_network() -> None:
    transport = UrllibTransport()

    with pytest.raises(K3XError, match="UNTRUSTED_OFFICIAL_HOST"):
        transport.get(
            "https://example.com/not-requested",
            headers={},
            max_bytes=1,
            timeout_seconds=2.0,
        )
