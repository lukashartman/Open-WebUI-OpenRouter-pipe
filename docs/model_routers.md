# Router-Class Models

Some OpenRouter models are **routers**: a single model id that selects a different
downstream model — and sometimes a different reasoning-effort level — for each request.
`typesafe/jev-router` and `openrouter/auto` are the best-known examples. The pipe treats
these as a distinct class because several of its normal assumptions do not hold for them.

> **Quick navigation:** [Docs Home](README.md) · [ZDR](openrouter_zdr.md) · [Provider Routing](openrouter_provider_routing.md) · [Valves](valves_and_configuration_atlas.md)

## Which models are routers

The pipe derives the class at catalog load from two signals:

- a negative `pricing.prompt` (the "selection step only, billed downstream" sentinel), or
- `architecture.tokenizer == "Router"` with `pricing.prompt == 0` and `pricing.completion == 0`.

The second clause catches `openrouter/free`. `~provider/model-latest` aliases are **not**
routers: they carry real, small non-zero pricing despite also using the router tokenizer.
The complete set at the time of writing is `typesafe/jev-router`, `nvidia/switchyard`,
`openrouter/auto`, `openrouter/auto-beta`, `openrouter/free`, `openrouter/fusion`,
`openrouter/pareto-code`, and `openrouter/bodybuilder`.

## Customizing the router's selection

Two admin valves constrain what a hosted router is allowed to pick:

- `Router allowed models` (`ROUTER_ALLOWED_MODELS`) — only models matching one of these
  patterns are candidates. Empty means the router's full pool.
- `Router excluded models` (`ROUTER_EXCLUDED_MODELS`) — matching models are never picked.

Both take comma-separated patterns, matched case-sensitively upstream: an exact slug
(`openai/gpt-5.1`), a wildcard (`anthropic/*`, `openai/gpt-5*`, `*flash*`), or a
`~author/family-latest` alias. Up to 1024 entries per list.

The pipe attaches them as a `plugins` entry on the request, under the plugin id the router
expects:

```json
{"plugins": [{"id": "jev-router", "allowed_models": ["anthropic/*"], "excluded_models": ["openai/gpt-4o"]}]}
```

The ids are `jev-router` for `typesafe/jev-router`, `auto-router` for `openrouter/auto`,
and `auto-beta-router` for `openrouter/auto-beta`; every other model is sent untouched.
With both valves empty, no entry is sent.

The lists only narrow the router's candidate pool — they never add models and do not change
how the router ranks what remains. Two upstream behaviours matter when writing them:
an allow list that matches nothing at all is ignored (the router falls back to its full
pool, still honouring exclusions), while exclusions are never ignored — if they remove
every candidate the request fails with OpenRouter `404`. These valves govern the router's
internal choice only; they do not change catalog visibility or ZDR handling.

## Zero Data Retention

Routers expose **no provider endpoints**, so they never appear in OpenRouter's
`/endpoints/zdr` list. Both ZDR controls therefore treat them as non-ZDR by default:
`Show only ZDR models` hides them and `Enforce ZDR routing` rejects them.

To admit a router, list it in `Admit routers through ZDR` (`ZDR_ROUTER_MODELS`). A listed
router stays selectable and, under enforcement, its request is sent with
`provider.zdr=true`.

**The caveat is real:** the ZDR guarantee covers the routing step only. The downstream
model the router picks is a separate data path that OpenRouter does not promise is ZDR.
Leaving the valve empty keeps the conservative default.

## Reasoning

The router chooses its own reasoning effort. The pipe asks for reasoning traces
(`summary` and `enabled`) but does not inject a fixed `effort` for router models, so the
router's own selection is not overridden. If a request explicitly sets an effort — through
the `reasoning_effort` transform or a per-chat control — that choice is preserved.

## Pricing and billing

A router's catalog price is a selection-step sentinel, not the price you pay. Real cost
comes from the response's `usage.cost`, which the pipe surfaces as usual. The negative
sentinel is never summed into a "free" verdict, so a router is not mislabelled as free;
zero-priced routers such as `openrouter/free` are still classified as free.

## Provider routing

Router models return an empty `/endpoints` payload, so there are no providers to populate
a routing dropdown. When a router is named in `ADMIN_PROVIDER_ROUTING_MODELS` or
`USER_PROVIDER_ROUTING_MODELS`, the pipe skips it, logs one warning, and generates no
filter. Ask OpenRouter to route the router instead.

## Observability

When a request to a router comes back with a different upstream `model` id, the pipe emits
one best-effort status line — `Routed via <id>` — and a debug log. It does not change the
Open WebUI `model` field, and it makes no assumptions about routing metadata: this is a
convenience, not a contract.
