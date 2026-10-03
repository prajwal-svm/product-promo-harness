import json

import httpx
import pytest
import respx

from promo.providers import cache, costs
from promo.providers.http import ProviderError, request

URL = "https://api.example.test/v1/thing"


@respx.mock
def test_retries_429_honouring_retry_after():
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json={"ok": True}),
    ])
    waits = []
    resp = request("demo", "POST", URL, json={}, sleep=waits.append)
    assert resp.json() == {"ok": True}
    assert route.call_count == 2 and waits == [2.0]


@respx.mock
def test_5xx_exhausts_retries():
    respx.get(URL).mock(return_value=httpx.Response(503, text="down"))
    waits = []
    with pytest.raises(ProviderError) as err:
        request("demo", "GET", URL, retries=2, backoff=1.0, sleep=waits.append)
    assert err.value.status == 503 and waits == [1.0, 2.0]


@respx.mock
def test_4xx_raises_with_provider_message_and_no_secret():
    respx.post(URL).mock(return_value=httpx.Response(401, json={"error": {"message": "API key not valid"}}))
    with pytest.raises(ProviderError) as err:
        request("gemini", "POST", URL, headers={"x-goog-api-key": "SECRET123"}, json={})
    msg = str(err.value)
    assert "gemini: HTTP 401: API key not valid" in msg
    assert "SECRET123" not in msg


@respx.mock
def test_detail_message_shapes():
    respx.post(URL).mock(return_value=httpx.Response(422, json={"detail": {"message": "bad plan"}}))
    with pytest.raises(ProviderError, match="bad plan"):
        request("elevenlabs", "POST", URL, json={})


@respx.mock
def test_network_error_retried():
    respx.get(URL).mock(side_effect=[httpx.ConnectError("boom"), httpx.Response(200, text="ok")])
    waits = []
    assert request("demo", "GET", URL, sleep=waits.append).text == "ok"
    assert len(waits) == 1


@respx.mock
def test_retries_all_5xx_including_524():
    route = respx.get(URL).mock(side_effect=[
        httpx.Response(524, text="down"),
        httpx.Response(200, json={"ok": True}),
    ])
    waits = []
    resp = request("demo", "GET", URL, sleep=waits.append)
    assert resp.json() == {"ok": True}
    assert route.call_count == 2 and len(waits) == 1


@respx.mock
def test_post_with_read_timeout_raises_immediately():
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("timeout"))
    waits = []
    with pytest.raises(ProviderError) as err:
        request("demo", "POST", URL, json={}, sleep=waits.append)
    assert err.value.status == 0 and len(waits) == 0


@respx.mock
def test_post_with_connect_error_retried():
    respx.post(URL).mock(side_effect=[
        httpx.ConnectError("boom"),
        httpx.Response(200, json={"ok": True}),
    ])
    waits = []
    resp = request("demo", "POST", URL, json={}, sleep=waits.append)
    assert resp.json() == {"ok": True}
    assert len(waits) == 1


def _google_429(message, *, retry=None, quota_id="GenerateRequestsPerMinutePerProjectPerModel", value="10"):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_requests_per_model",
                                "quotaId": quota_id, "quotaValue": value}]}]
    if retry:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry})
    return httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": message,
                                               "details": details}})


@respx.mock
def test_google_429_waits_the_retry_delay_it_names():
    route = respx.post(URL).mock(side_effect=[
        _google_429("You exceeded your current quota ... Please retry in 44.31s.", retry="44s"),
        _google_429("You exceeded your current quota ... Please retry in 3.5s."),     # no RetryInfo: read the text
        httpx.Response(200, json={"ok": True}),
    ])
    waits = []
    assert request("gemini", "POST", URL, json={}, sleep=waits.append).json() == {"ok": True}
    assert route.call_count == 3 and waits == [44.0, 3.5]


@respx.mock
def test_429_waits_are_bounded():
    # a Retry-After header past a minute is capped; a Google RetryInfo past a minute fails at once (test below)
    route = respx.post(URL).mock(return_value=httpx.Response(429, headers={"retry-after": "900"},
                                                             json={"error": {"message": "slow down"}}))
    waits = []
    with pytest.raises(ProviderError) as err:
        request("gemini", "POST", URL, json={}, retries=2, sleep=waits.append)
    assert err.value.status == 429 and waits == [60.0, 60.0] and route.call_count == 3


@pytest.mark.parametrize("quota_id, value, message", [
    ("GenerateRequestsPerDayPerProjectPerModel", "50", "You exceeded your current quota, please check your plan and "
                                                      "billing details. limit: 50, model: lyria-3.5"),
    ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "0", "Quota exceeded ... limit: 0, model: lyria-3.5"),
])
@respx.mock
def test_a_quota_that_will_not_reset_soon_fails_at_once_and_says_so(quota_id, value, message):
    route = respx.post(URL).mock(return_value=_google_429(message, retry="30s", quota_id=quota_id, value=value))
    waits = []
    with pytest.raises(ProviderError) as err:
        request("lyria", "POST", URL, json={}, sleep=waits.append)
    assert route.call_count == 1 and waits == []
    assert err.value.status == 429 and "will not reset within minutes" in str(err.value)
    assert "plan" in str(err.value) and message.split(".")[0] in str(err.value)


@respx.mock
def test_a_named_wait_longer_than_a_retry_can_wait_fails_at_once_and_quotes_it():
    # a daily cap whose quota id does not say PerDay: the 14-hour RetryInfo alone gives it away
    route = respx.post(URL).mock(return_value=_google_429("You exceeded your current quota.", retry="51180s"))
    waits = []
    with pytest.raises(ProviderError) as err:
        request("lyria", "POST", URL, json={}, sleep=waits.append)
    assert route.call_count == 1 and waits == []
    assert "retry in 14h13m" in str(err.value) and "will not reset within minutes" in str(err.value)


@respx.mock
def test_a_per_day_metric_in_the_message_alone_is_a_daily_quota():
    msg = ("Quota exceeded for metric: generativelanguage.googleapis.com/generate_requests_per_model_per_day, "
           "limit: 100, model: gemini-3.8-flash-tts. Please retry in 31.2s.")
    route = respx.post(URL).mock(return_value=httpx.Response(429, json={"error": {"code": 429, "message": msg}}))
    with pytest.raises(ProviderError) as err:
        request("gemini", "POST", URL, json={}, sleep=lambda s: None)
    assert route.call_count == 1 and "daily quota; retry in 31s" in str(err.value)


@respx.mock
def test_retry_after_nan_falls_back_to_backoff():
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(429, headers={"retry-after": "nan"}, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json={"ok": True}),
    ])
    waits = []
    resp = request("demo", "POST", URL, json={}, backoff=1.5, sleep=waits.append)
    assert resp.json() == {"ok": True}
    assert route.call_count == 2 and waits == [1.5]


def test_cache_round_trip():
    assert cache.cache_get("demo", "abc123") is None
    cache.cache_put("demo", "abc123", b"hello")
    assert cache.cache_get("demo", "abc123") == b"hello"


def test_cache_hit_skips_http():
    calls = []

    def fn():
        calls.append(1)
        return b"paid-result"

    first, hit1 = cache.cached_call("demo", "key1", fn)
    second, hit2 = cache.cached_call("demo", "key1", fn)
    assert first == second == b"paid-result"
    assert (hit1, hit2) == (False, True)
    assert len(calls) == 1


def test_cost_log(tmp_path):
    log = tmp_path / "costs.jsonl"
    costs.log_cost(log, provider="gemini", model="m", units={"audio_s": 10}, usd=0.0023, cached=False)
    costs.log_cost(log, provider="gemini", model="m", units={"audio_s": 10}, usd=0.0023, cached=True)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert rows[0]["usd"] == 0.0023 and rows[1]["cached"] is True
    assert costs.total_usd(log) == pytest.approx(0.0023)  # cached rows cost nothing
    costs.log_cost(None, provider="x", model="y", units={}, usd=1.0, cached=False)  # no run: no-op
