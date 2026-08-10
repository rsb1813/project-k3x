# 공식 가중치 HTTP 요청의 호스트와 바이트 한도를 강제합니다.
from __future__ import annotations

from dataclasses import dataclass
from email.message import Message
from types import MappingProxyType
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .format import K3XError


@dataclass(frozen=True)
class HttpResponse:
    status: int
    final_url: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class TransportStats:
    requests: int
    response_bytes: int
    maximum_response_bytes: int


def _is_official_host(host: str) -> bool:
    value = host.lower().rstrip(".")
    return value == "huggingface.co" or value.endswith(".hf.co")


class _ValidatingRedirectHandler(HTTPRedirectHandler):
    def __init__(self, transport: "UrllibTransport") -> None:
        super().__init__()
        self._transport = transport
        self._redirects = 0

    def redirect_request(
        self,
        req: Request,
        fp,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> Request | None:
        self._redirects += 1
        self._transport._requests += 1
        if self._redirects > self._transport.max_redirects:
            raise K3XError("OFFICIAL_REDIRECT_LIMIT")
        resolved = urljoin(req.full_url, newurl)
        self._transport._validate_url(resolved)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


class UrllibTransport:
    def __init__(
        self,
        *,
        max_redirects: int = 5,
        allowed_hosts: frozenset[str] | None = None,
    ) -> None:
        if max_redirects < 0:
            raise ValueError("max_redirects must be non-negative")
        self.max_redirects = max_redirects
        self._allowed_hosts = (
            frozenset(host.lower() for host in allowed_hosts)
            if allowed_hosts is not None
            else None
        )
        self._requests = 0
        self._response_bytes = 0
        self._maximum_response_bytes = 0

    @property
    def stats(self) -> TransportStats:
        return TransportStats(
            self._requests, self._response_bytes, self._maximum_response_bytes
        )

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if self._allowed_hosts is None:
            valid = parsed.scheme == "https" and _is_official_host(host)
        else:
            valid = parsed.scheme in {"http", "https"} and host in self._allowed_hosts
        if not valid or parsed.username is not None or parsed.password is not None:
            raise K3XError("UNTRUSTED_OFFICIAL_HOST")

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
        timeout_seconds: float,
        expected_status: int = 200,
    ) -> HttpResponse:
        if max_bytes < 0 or timeout_seconds <= 0:
            raise ValueError("invalid HTTP bound")
        self._validate_url(url)
        handler = _ValidatingRedirectHandler(self)
        opener = build_opener(handler)
        request = Request(url, headers=dict(headers), method="GET")
        self._requests += 1
        try:
            response = opener.open(request, timeout=timeout_seconds)
        except K3XError:
            raise
        except HTTPError as error:
            raise K3XError("OFFICIAL_HTTP_STATUS", str(error.code)) from error
        except (OSError, URLError) as error:
            raise K3XError("OFFICIAL_HTTP_ERROR") from error
        with response:
            status = response.getcode()
            if status != expected_status:
                raise K3XError("OFFICIAL_HTTP_STATUS", str(status))
            self._validate_url(response.geturl())
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError as error:
                    raise K3XError("INVALID_OFFICIAL_HEADER") from error
                if declared_bytes < 0 or declared_bytes > max_bytes:
                    raise K3XError("OFFICIAL_BODY_LIMIT")
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise K3XError("OFFICIAL_BODY_LIMIT")
            normalized = MappingProxyType(
                {key.lower(): value.strip() for key, value in response.headers.items()}
            )
            self._response_bytes += len(body)
            self._maximum_response_bytes = max(self._maximum_response_bytes, len(body))
            return HttpResponse(status, response.geturl(), normalized, body)

