"""Router-class model support: detection, ZDR opt-in, reasoning, pricing,
provider routing, and observability."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from aioresponses import aioresponses
from aioresponses.core import CallbackResult

from open_webui_openrouter_pipe.api.transforms import ResponsesBody
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
from open_webui_openrouter_pipe.pipe import EncryptedStr, Pipe


class _DummyResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200, reason: str = "OK"):
        self._payload = payload
        self.status = status
        self.reason = reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    async def text(self):
        return json.dumps(self._payload)

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status} {self.reason}")

    @property
    def headers(self):
        return {}


class _CatalogSession:
    """Return the model catalog for /models and the given ZDR list otherwise."""

    def __init__(self, models: list[dict[str, Any]], zdr: list[dict[str, Any]] | None = None):
        self._models = models
        self._zdr = zdr or []

    def get(self, url, *_, **__):
        if str(url).endswith("/endpoints/zdr"):
            return _DummyResponse({"data": self._zdr})
        return _DummyResponse({"data": self._models})


ROUTER_MODEL = {
    "id": "typesafe/jev-router",
    "name": "Jev Router",
    "context_length": 1000000,
    "pricing": {"prompt": "-1", "completion": "-1"},
    "architecture": {
        "tokenizer": "Router",
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
    },
    "supported_parameters": ["reasoning", "tools"],
}

ZERO_PRICED_ROUTER = {
    "id": "openrouter/free",
    "name": "Free Router",
    "pricing": {"prompt": "0", "completion": "0"},
    "architecture": {"tokenizer": "Router", "output_modalities": ["text"]},
    "supported_parameters": ["reasoning"],
}

TILDE_ALIAS = {
    "id": "~anthropic/claude-opus-latest",
    "name": "Claude Opus (latest)",
    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
    "architecture": {"tokenizer": "Router", "output_modalities": ["text"]},
    "supported_parameters": ["reasoning"],
}

CONCRETE_MODEL = {
    "id": "openai/gpt-5",
    "name": "GPT-5",
    "pricing": {"prompt": "0.001", "completion": "0.004"},
    "architecture": {"tokenizer": "GPT", "output_modalities": ["text"]},
    "supported_parameters": ["reasoning", "tools"],
}

FREE_CONCRETE_MODEL = {
    "id": "meta-llama/llama-free",
    "name": "Llama Free",
    "pricing": {"prompt": "0", "completion": "0"},
    "architecture": {"tokenizer": "Llama", "output_modalities": ["text"]},
    "supported_parameters": ["tools"],
}

NO_PRICING_MODEL = {
    "id": "example/no-pricing",
    "name": "No Pricing",
    "architecture": {"tokenizer": "Router", "output_modalities": ["text"]},
    "supported_parameters": ["reasoning"],
}


async def _load_catalog(session=None, zdr=None) -> None:
    session = session or _CatalogSession(
        [ROUTER_MODEL, ZERO_PRICED_ROUTER, TILDE_ALIAS, CONCRETE_MODEL,
         FREE_CONCRETE_MODEL, NO_PRICING_MODEL],
        zdr=zdr,
    )
    await OpenRouterModelRegistry.ensure_loaded(
        session,
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        cache_seconds=3600,
        logger=logging.getLogger("test"),
    )


@pytest.mark.asyncio
async def test_negative_pricing_is_router():
    await _load_catalog()
    assert OpenRouterModelRegistry.is_router_model("typesafe/jev-router") is True


@pytest.mark.asyncio
async def test_zero_priced_router_tokenizer_is_router():
    await _load_catalog()
    assert OpenRouterModelRegistry.is_router_model("openrouter/free") is True


@pytest.mark.asyncio
async def test_zero_priced_concrete_model_is_not_router():
    """Review Focus 3: tokenizer must be Router, not merely free pricing."""
    await _load_catalog()
    assert OpenRouterModelRegistry.is_router_model("meta-llama/llama-free") is False


@pytest.mark.asyncio
async def test_tilde_latest_alias_is_not_router():
    await _load_catalog()
    assert OpenRouterModelRegistry.is_router_model("~anthropic/claude-opus-latest") is False


@pytest.mark.asyncio
async def test_concrete_model_is_not_router():
    await _load_catalog()
    assert OpenRouterModelRegistry.is_router_model("openai/gpt-5") is False


@pytest.mark.asyncio
async def test_missing_pricing_is_not_router():
    """Review Focus 4: absent pricing must be False, not a crash."""
    await _load_catalog()
    assert OpenRouterModelRegistry.is_router_model("example/no-pricing") is False


@pytest.mark.asyncio
async def test_variant_suffix_resolves_to_base_router():
    """Review Focus 1: a :variant of a router is still a router."""
    await _load_catalog()
    assert OpenRouterModelRegistry.is_router_model("typesafe/jev-router:nitro") is True


def test_unknown_id_is_not_router():
    assert OpenRouterModelRegistry.is_router_model("nobody/unknown") is False
    assert OpenRouterModelRegistry.is_router_model("") is False


@pytest.mark.asyncio
async def test_negative_priced_router_is_not_free():
    await _load_catalog()
    from open_webui_openrouter_pipe.models.registry import is_free_model

    assert is_free_model("typesafe.jev-router") is False


@pytest.mark.asyncio
async def test_zero_priced_router_still_free():
    await _load_catalog()
    from open_webui_openrouter_pipe.models.registry import is_free_model

    assert is_free_model("openrouter.free") is True


@pytest.mark.asyncio
async def test_missing_pricing_is_not_free():
    """Review Focus 4: absent pricing is False, not a crash."""
    await _load_catalog()
    from open_webui_openrouter_pipe.models.registry import is_free_model

    assert is_free_model("example.no-pricing") is False


def _router_model_rows():
    return [
        {"id": "typesafe.jev-router", "norm_id": "typesafe.jev-router",
         "name": "Jev Router", "original_id": "typesafe/jev-router"},
        {"id": "openai.gpt-5", "norm_id": "openai.gpt-5",
         "name": "GPT-5", "original_id": "openai/gpt-5"},
    ]


# gpt-5 is ZDR-capable, so it survives the filter and the router's treatment is isolated.
_ZDR_LIST = [{"model_id": "openai/gpt-5"}]


@pytest.mark.asyncio
async def test_router_hidden_under_zdr_only_without_opt_in():
    await _load_catalog(zdr=_ZDR_LIST)
    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    pipe.valves.ZDR_ROUTER_MODELS = ""
    kept = pipe._apply_model_filters(_router_model_rows(), pipe.valves)
    assert [m["norm_id"] for m in kept] == ["openai.gpt-5"]


@pytest.mark.asyncio
async def test_listed_router_visible_under_zdr_only():
    await _load_catalog(zdr=_ZDR_LIST)
    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    pipe.valves.ZDR_ROUTER_MODELS = "typesafe/jev-router"
    kept = pipe._apply_model_filters(_router_model_rows(), pipe.valves)
    assert [m["norm_id"] for m in kept] == ["typesafe.jev-router", "openai.gpt-5"]


@pytest.mark.asyncio
async def test_listed_router_variant_resolves_to_base():
    """Review Focus 1: a :variant of a listed router is admitted."""
    await _load_catalog(zdr=_ZDR_LIST)
    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    pipe.valves.ZDR_ROUTER_MODELS = "typesafe/jev-router"
    rows = [
        {"id": "typesafe.jev-router:exacto", "norm_id": "typesafe.jev-router:exacto",
         "name": "Jev Router Exacto", "original_id": "typesafe/jev-router",
         "variant_is_virtual": True, "variant_base_norm_id": "typesafe.jev-router"},
    ]
    kept = pipe._apply_model_filters(rows, pipe.valves)
    assert [m["norm_id"] for m in kept] == ["typesafe.jev-router:exacto"]


@pytest.mark.asyncio
async def test_unknown_valve_id_does_not_admit_router():
    """Review Focus 5: a typo is inert."""
    await _load_catalog(zdr=_ZDR_LIST)
    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    pipe.valves.ZDR_ROUTER_MODELS = "typesafe/jev-routr"
    kept = pipe._apply_model_filters(_router_model_rows(), pipe.valves)
    assert [m["norm_id"] for m in kept] == ["openai.gpt-5"]


@pytest.mark.asyncio
async def test_restriction_reason_omitted_for_listed_router():
    await _load_catalog()
    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    pipe.valves.ZDR_ROUTER_MODELS = "typesafe/jev-router"
    reasons = pipe._model_restriction_reasons(
        "typesafe.jev-router",
        valves=pipe.valves,
        allowlist_norm_ids={"typesafe.jev-router"},
        catalog_norm_ids={"typesafe.jev-router", "openai.gpt-5"},
    )
    assert "ZDR_MODELS_ONLY" not in reasons


@pytest.mark.asyncio
async def test_restriction_reason_present_for_unlisted_router():
    await _load_catalog()
    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    pipe.valves.ZDR_ROUTER_MODELS = ""
    reasons = pipe._model_restriction_reasons(
        "typesafe.jev-router",
        valves=pipe.valves,
        allowlist_norm_ids={"typesafe.jev-router"},
        catalog_norm_ids={"typesafe.jev-router", "openai.gpt-5"},
    )
    assert "ZDR_MODELS_ONLY" in reasons


_ROUTER_SSE = (
    'data: {"type":"response.output_text.delta","delta":"OK"}\n\n'
    'data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":5,"output_tokens":3}}}\n\n'
)


async def _consume(result) -> None:
    if hasattr(result, "__aiter__"):
        async for _chunk in result:
            pass


async def _run_zdr_request(valve_router_ids: str):
    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
        pipe.valves.ZDR_ENFORCE = True
        pipe.valves.ZDR_ROUTER_MODELS = valve_router_ids

        captured: list[dict] = []

        def callback(url, **kwargs):
            captured.append(kwargs.get("json", {}))
            return CallbackResult(
                body=_ROUTER_SSE.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
                status=200,
            )

        async def event_emitter(event):
            pass

        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", callback=callback, repeat=True)
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [ROUTER_MODEL]},
                repeat=True,
            )
            mock_http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr",
                payload={"data": []},
                repeat=True,
            )
            result = await pipe.pipe(
                body={"model": "typesafe/jev-router",
                      "messages": [{"role": "user", "content": "hi"}], "stream": True},
                __user__={"id": "user_123"},
                __request__=None,
                __event_emitter__=event_emitter,
                __event_call__=None,
                __metadata__={"model": {"id": "typesafe/jev-router"}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
            await _consume(result)
        return captured
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_enforce_zdr_admits_listed_router():
    captured = await _run_zdr_request("typesafe/jev-router")
    assert captured, "Expected the router request to be sent"
    assert (captured[-1].get("provider") or {}).get("zdr") is True


@pytest.mark.asyncio
async def test_enforce_zdr_rejects_unlisted_router():
    captured = await _run_zdr_request("")
    assert captured == [], "An unlisted router must be rejected before any request is sent"


@pytest.mark.asyncio
async def test_router_receives_no_injected_effort():
    await _load_catalog()
    pipe = Pipe()
    pipe.valves.ENABLE_REASONING = True
    pipe.valves.REASONING_EFFORT = "medium"
    pipe.valves.REASONING_SUMMARY_MODE = "auto"
    mgr = pipe._ensure_reasoning_config_manager()

    body = ResponsesBody(model="typesafe/jev-router", input=[])
    mgr._apply_reasoning_preferences(body, pipe.valves)

    assert isinstance(body.reasoning, dict)
    assert "effort" not in body.reasoning
    assert body.reasoning.get("summary") == "auto"
    assert body.reasoning.get("enabled") is True


@pytest.mark.asyncio
async def test_router_preserves_explicit_effort():
    await _load_catalog()
    pipe = Pipe()
    pipe.valves.ENABLE_REASONING = True
    pipe.valves.REASONING_EFFORT = "medium"
    mgr = pipe._ensure_reasoning_config_manager()

    body = ResponsesBody(model="typesafe/jev-router", reasoning={"effort": "high"}, input=[])
    mgr._apply_reasoning_preferences(body, pipe.valves)

    assert body.reasoning.get("effort") == "high"


@pytest.mark.asyncio
async def test_non_router_still_receives_valve_effort():
    await _load_catalog()
    pipe = Pipe()
    pipe.valves.ENABLE_REASONING = True
    pipe.valves.REASONING_EFFORT = "medium"
    mgr = pipe._ensure_reasoning_config_manager()

    body = ResponsesBody(model="openai/gpt-5", input=[])
    mgr._apply_reasoning_preferences(body, pipe.valves)

    assert body.reasoning.get("effort") == "medium"


@pytest.mark.asyncio
async def test_task_reasoning_leaves_router_effort_unset():
    await _load_catalog()
    pipe = Pipe()
    mgr = pipe._ensure_reasoning_config_manager()

    body = ResponsesBody(model="typesafe/jev-router", input=[])
    mgr._apply_task_reasoning_preferences(body, "high")

    assert "effort" not in (body.reasoning or {})
    assert (body.reasoning or {}).get("enabled") is True


from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_router_is_skipped_in_provider_overlay(pipe_instance_async):
    """A router in a routing valve must not trigger an endpoints fetch."""
    pipe = pipe_instance_async
    OpenRouterModelRegistry._specs["typesafe.jev-router"] = {"is_router": True}
    manager = pipe._ensure_catalog_manager()

    with patch.object(manager, "_fetch_model_endpoints", new_callable=AsyncMock) as fetch:
        overlay = await manager._build_routed_provider_overlay(
            None, ["typesafe/jev-router"]
        )

    fetch.assert_not_awaited()
    assert overlay == {}


from open_webui_openrouter_pipe.models.registry import router_downstream_model_id


@pytest.mark.asyncio
async def test_downstream_model_reported_for_router():
    await _load_catalog()
    assert (
        router_downstream_model_id(
            "typesafe/jev-router", {"model": "anthropic/claude-sonnet-4.6"}
        )
        == "anthropic/claude-sonnet-4.6"
    )


@pytest.mark.asyncio
async def test_downstream_model_none_when_unchanged():
    await _load_catalog()
    assert router_downstream_model_id("typesafe/jev-router", {"model": "typesafe/jev-router"}) is None


@pytest.mark.asyncio
async def test_downstream_model_none_for_non_router():
    await _load_catalog()
    assert router_downstream_model_id("openai/gpt-5", {"model": "openai/gpt-5"}) is None


@pytest.mark.asyncio
async def test_downstream_model_none_for_missing_field():
    await _load_catalog()
    assert router_downstream_model_id("typesafe/jev-router", {}) is None
    assert router_downstream_model_id("typesafe/jev-router", None) is None
    assert router_downstream_model_id("typesafe/jev-router", {"model": "  "}) is None


@pytest.mark.asyncio
async def test_router_route_status_emitted_once():
    """The streaming loop emits exactly one best-effort status line."""
    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"

        sse = (
            'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
            'data: {"type":"response.completed","response":'
            '{"model":"anthropic/claude-sonnet-4.6","output":[],'
            '"usage":{"input_tokens":5,"output_tokens":3}}}\n\n'
        )

        def callback(url, **kwargs):
            return CallbackResult(
                body=sse.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
                status=200,
            )

        events: list[dict] = []

        async def event_emitter(event):
            events.append(event)

        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", callback=callback, repeat=True)
            mock_http.get(
                "https://openrouter.ai/api/v1/models", payload={"data": [ROUTER_MODEL]}, repeat=True
            )
            mock_http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr", payload={"data": []}, repeat=True
            )
            result = await pipe.pipe(
                body={"model": "typesafe/jev-router",
                      "messages": [{"role": "user", "content": "hi"}], "stream": True},
                __user__={"id": "user_123"},
                __request__=None,
                __event_emitter__=event_emitter,
                __event_call__=None,
                __metadata__={"model": {"id": "typesafe/jev-router"}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
            # Streaming requests are delivered through the middleware stream
            # channel (chunks of {"event": {...}}), not the raw emitter.
            if hasattr(result, "__aiter__"):
                async for _chunk in result:
                    if isinstance(_chunk, dict) and isinstance(_chunk.get("event"), dict):
                        events.append(_chunk["event"])

        routed = [
            e for e in events
            if isinstance(e, dict)
            and e.get("type") == "status"
            and "Routed via" in str((e.get("data") or {}).get("description", ""))
        ]
        assert len(routed) == 1, f"expected exactly one route status, got {routed}"
        assert "anthropic/claude-sonnet-4.6" in routed[0]["data"]["description"]
    finally:
        await pipe.close()
