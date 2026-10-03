"""HTTP with retries and readable provider errors. Never echoes request headers."""

from __future__ import annotations

import math
import re
import time

import httpx

MAX_WAIT_S = 60.0


class ProviderError(RuntimeError):
    def __init__(self, provider: str, status: int, message: str):
        super().__init__(f"{provider}: HTTP {status}: {message}")
        self.provider = provider
        self.status = status
        self.detail = message


def _message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300] or resp.reason_phrase
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        detail = data.get("detail")
        if isinstance(detail, dict) and detail.get("message"):
            return str(detail["message"])
        if isinstance(detail, str):
            return detail
        if isinstance(err, str):
            return err
    return str(data)[:300]


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after", "")
    try:
        parsed = float(value)
        # Ignore NaN, infinite, or negative values
        if math.isnan(parsed) or math.isinf(parsed) or parsed < 0:
            return None
        return min(parsed, MAX_WAIT_S)
    except ValueError:
        return _google_quota(resp)[0]


def _span(seconds: float) -> str:
    """44 -> "44s", 754 -> "12m34s", 51180 -> "14h13m"."""
    seconds = int(round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _google_quota(resp: httpx.Response) -> tuple[float | None, str | None]:
    """A Google API 429 (RESOURCE_EXHAUSTED): (the wait it names, capped at MAX_WAIT_S; why the quota will stay
    exhausted for longer than a retry can wait, or None). The wait comes from a RetryInfo detail ("44s"), else from
    "retry in 44.3s" in the message. Persistent: a per-day quota (a PerDay quota id, or per_day in the message), a
    quota of 0 (the model is not on this key's plan), or a named wait longer than MAX_WAIT_S."""
    try:
        data = resp.json()
    except ValueError:
        return None, None
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return None, None
    message = str(err.get("message", ""))
    wait, daily, none_left = None, bool(re.search(r"per_?day", message, re.I)), bool(re.search(r"\blimit: 0\b", message))
    for d in err.get("details") or []:
        if not isinstance(d, dict):
            continue
        kind = str(d.get("@type", ""))
        if kind.endswith("RetryInfo"):
            m = re.fullmatch(r"\s*([\d.]+)s\s*", str(d.get("retryDelay", "")))
            wait = float(m.group(1)) if m else wait
        elif kind.endswith("QuotaFailure"):
            for v in d.get("violations") or []:
                if isinstance(v, dict):
                    daily = daily or "PerDay" in str(v.get("quotaId", ""))
                    none_left = none_left or str(v.get("quotaValue")) == "0"
    if wait is None:
        m = re.search(r"retry in ([\d.]+)\s*s", message)
        wait = float(m.group(1)) if m else None
    why = None
    if none_left:
        why = "a quota of 0: this key's plan does not include the model"
    elif daily or (wait is not None and wait > MAX_WAIT_S):
        why = ("daily quota" if daily else "quota") + (f"; retry in {_span(wait)}" if wait is not None else "")
    return (min(wait, MAX_WAIT_S) if wait is not None else None), why


def request(
    provider: str,
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    json: dict | None = None,
    data: dict | None = None,
    files: dict | None = None,
    params: dict | None = None,
    timeout: float = 180.0,
    retries: int = 3,
    backoff: float = 1.5,
    client: httpx.Client | None = None,
    sleep=time.sleep,
) -> httpx.Response:
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    attempt = 0
    try:
        while True:
            try:
                resp = client.request(method, url, headers=headers, json=json, data=data, files=files,
                                      params=params, timeout=timeout)
            except httpx.TransportError as e:
                # Only retry ConnectError and ConnectTimeout for any method.
                # Retry other TransportErrors only on GET (to avoid double-billing on POST/PUT/PATCH).
                should_retry = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)) or (method == "GET")
                if not should_retry or attempt >= retries:
                    raise ProviderError(provider, 0, f"network error: {e}") from e
                sleep(backoff * (2 ** attempt))
                attempt += 1
                continue
            if resp.status_code == 429 and (why := _google_quota(resp)[1]):
                raise ProviderError(provider, 429, f"quota exhausted, and it will not reset within minutes ({why}; "
                                    f"check the plan and billing): " + _message(resp))
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt < retries:
                wait = _retry_after(resp)
                sleep(wait if wait is not None else backoff * (2 ** attempt))
                attempt += 1
                continue
            if resp.status_code >= 400:
                raise ProviderError(provider, resp.status_code, _message(resp))
            return resp
    finally:
        if own:
            client.close()
