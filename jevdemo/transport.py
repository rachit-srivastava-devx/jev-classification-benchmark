"""The only module in this package that touches the network.

Everything else takes a `Transport` callable, so the whole test suite runs against
captured fixtures with no key and no spend. `test_transport.py` asserts that
property by grepping the package rather than trusting it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

#: (url, body, headers) -> (status, raw_body, wall_ms)
Transport = Callable[[str, dict, dict], tuple[int, bytes, float]]

DEFAULT_TIMEOUT_SECONDS = 60.0


class TransportError(RuntimeError):
    """The request never reached a server, or its reply never arrived."""


@dataclass(frozen=True)
class RecordedCall:
    """One request a `FixtureTransport` received.

    The Authorization header is deliberately absent: recorded calls end up in test
    output and failure messages, and a bearer token must not travel with them.
    """

    url: str
    body: dict
    header_names: tuple[str, ...]


class _Clock(Protocol):
    def __call__(self) -> float: ...


class HttpTransport:
    """POST JSON over HTTPS, timing only the request itself.

    No retries. A retry is a second billed call and a corrupted latency sample, so
    the one permitted retry in this system lives in the reasoning negotiation
    (spec 01), where it happens once per arm rather than once per ticket.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        clock: _Clock = time.perf_counter,
    ) -> None:
        self._timeout = timeout
        self._clock = clock

    def __call__(self, url: str, body: dict, headers: dict) -> tuple[int, bytes, float]:
        """Send one request.

        Args:
            url: Absolute https URL.
            body: JSON-serialisable request body.
            headers: Request headers, including Authorization.

        Returns:
            The status, the raw response body, and wall-clock milliseconds.

        Raises:
            TransportError: The request did not complete. An HTTP error status is
                not a transport error — it is returned like any other status, so
                the caller can read the body a 4xx carries.
        """
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = self._clock()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
                return response.status, raw, (self._clock() - started) * 1000.0
        except urllib.error.HTTPError as error:
            # A 400 carrying "reasoning is mandatory" is information, not a failure.
            return error.code, error.read(), (self._clock() - started) * 1000.0
        except OSError as error:
            # Broader than URLError on purpose: urllib wraps only the connect phase,
            # so a timeout while reading the response arrives as a bare TimeoutError.
            reason = getattr(error, "reason", error)
            raise TransportError(f"{type(error).__name__}: {reason}") from error


@dataclass
class FixtureTransport:
    """Replay captured responses. Used by every offline test.

    Args:
        responses: URL -> one `(status, body)` pair, or a list of them replayed in
            order. A list is what the reasoning negotiation needs: a 400, then a 200.
        wall_ms: The latency every replayed call reports.
    """

    responses: dict[str, object]
    wall_ms: float = 10.0
    calls: list[RecordedCall] = field(default_factory=list)
    _cursor: dict[str, int] = field(default_factory=dict)

    def __call__(self, url: str, body: dict, headers: dict) -> tuple[int, bytes, float]:
        """Replay the next response scripted for `url`.

        Raises:
            KeyError: `url` was never scripted. Returning an empty response instead
                would let a broken test pass.
            IndexError: The script for `url` is exhausted.
        """
        self.calls.append(
            RecordedCall(url=url, body=body, header_names=tuple(sorted(headers)))
        )
        if url not in self.responses:
            raise KeyError(f"no fixture scripted for {url}")

        scripted = self.responses[url]
        if isinstance(scripted, list):
            index = self._cursor.get(url, 0)
            if index >= len(scripted):
                raise IndexError(f"fixture script for {url} exhausted after {index} calls")
            self._cursor[url] = index + 1
            status, raw = scripted[index]
        else:
            status, raw = scripted  # type: ignore[misc]

        return status, raw, self.wall_ms


class Throttled:
    """Retries a 429 and nothing else.

    A 429 says our own request rate was too high; it is not evidence about the
    model, so recording it as a failure would put the rate limiter in the
    accuracy column. Every other status is returned untouched on the first
    attempt — a 500 retried is a second charge for the same broken answer.

    The latency returned is the successful attempt's own wall time, not the
    elapsed time across the retries. Timing the backoff would measure this
    script's patience.

    Not thread-safe in its `retries` counter; it is a report figure, not a
    control signal, and an occasional lost increment is acceptable there.
    """

    def __init__(self, inner: Transport, sleep=time.sleep, max_attempts: int = 5,
                 base_delay_s: float = 2.0):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._inner = inner
        self._sleep = sleep
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.retries = 0

    def __call__(self, url: str, body: dict, headers: dict) -> tuple[int, bytes, float]:
        for attempt in range(self.max_attempts):
            status, raw, wall_ms = self._inner(url, body, headers)
            if status != 429 or attempt == self.max_attempts - 1:
                return status, raw, wall_ms
            self.retries += 1
            self._sleep(self.base_delay_s * (2 ** attempt))
        raise AssertionError("unreachable: the loop returns on its last attempt")
