"""Router-class model support: detection, ZDR opt-in, reasoning, pricing,
provider routing, and observability."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry


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
