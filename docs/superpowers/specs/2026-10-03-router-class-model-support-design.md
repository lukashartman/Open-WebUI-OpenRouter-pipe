# Router-Class Model Support — Design

**Date:** 2026-10-03
**Status:** Approved design, pending written-spec review
**Scope:** ZDR filtering, reasoning shaping, pricing classification, provider routing, minimal observability for router-class OpenRouter models.

---

## 1. Context

OpenRouter serves **router-class** models: a single model id that dynamically selects a
downstream model (and, for `typesafe/jev-router`, a reasoning-effort level) per request.
The trigger for this work is `typesafe/jev-router` (released 2026-09-25), which behaves
like `openrouter/auto` for the purposes of this pipe.

The pipe already imports every catalog model, so routers are *listed* and *callable* with
no code change. But the pipe treats them as ordinary models, which produces several
misbehaviours. The one that was actually observed: **with `ZDR_MODELS_ONLY=true`, routers
are hidden from the model selector** — including `openrouter/auto`. Adding the model to
`MODEL_ID` does not help because the ZDR filter runs after `_select_models`.

### Root cause of the observed symptom

ZDR capability is derived from OpenRouter's per-model endpoints list
(`OpenRouterModelRegistry.is_zdr_capable`, `registry.py:942-958`, exact membership in the
set built from `/api/v1/endpoints/zdr`). Router-class models expose **zero endpoints**
(`/api/v1/models/typesafe/jev-router/endpoints` → `[]`; same for `openrouter/auto`) and are
absent from the ZDR list (only `typesafe/jev-1.13` appears there). So every router is
classified non-ZDR and dropped by `_apply_model_filters` (`pipe.py:3072-3077`). With
`ZDR_ENFORCE=true`, requests are likewise rejected (`orchestrator.py:680-733`).

This is conservative-safe by accident — a router's downstream model is a separate data path
that OpenRouter does not guarantee is ZDR — but it is silent and has no escape hatch.

### Reference

A vendor development guide for `typesafe/jev-router` was supplied
(`~/Downloads/jev-router-dev-guide.md`). Relevant points: pricing is a selection-step
sentinel with downstream billing; the Jev routing step is ZDR and `zdr:true` is supported,
but the downstream model is a separate data path; the router chooses reasoning effort
itself; routing metadata has no stable public JSON contract; there is no silent fallback on
a Jev failure.

---

## 2. Goal

Treat router-class models as a first-class class in the pipe so that:

1. They are not silently hidden by ZDR filters, with an explicit opt-in for operators who
   accept the downstream-data-path caveat.
2. The pipe does not override the router's own reasoning-effort selection.
3. Router sentinel pricing is classified correctly.
4. Provider-routing filters degrade gracefully for routers instead of producing empty
   dropdowns.
5. The downstream model OpenRouter reports back is surfaced, minimally.

### Non-goals

- No retry/pin/fallback policy for router failures. The pipe keeps its current behaviour
  (transport retries for network errors only; opt-in `model_fallback` custom parameter).
  A Jev failure continues to surface as an error.
- No structured parsing of routing metadata (the guide states there is no stable contract).
- No change to how `~…-latest` provider aliases or concrete models are handled.
- No change to fusion (`openrouter/fusion` is router-class by detection, but the pipe's
  fusion feature path is untouched; see §4 assumptions).

---

## 3. Definition: router-class detection

A model is **router-class** when either:

- `pricing.prompt < 0` (the "selection step only, billed downstream" sentinel), **or**
- `architecture.tokenizer == "Router"` **and** `pricing.prompt == 0` **and**
  `pricing.completion == 0`.

The second clause exists solely to catch `openrouter/free`, which is tokenizer `Router` with
`0/0` pricing. It does **not** re-introduce the false positives that a bare tokenizer match
would: every `~provider/model-latest` alias carries real, small non-zero pricing, and no
concrete model has tokenizer `Router`.

Resulting set at the time of writing (8 models):

```
typesafe/jev-router, nvidia/switchyard, openrouter/auto, openrouter/auto-beta,
openrouter/free, openrouter/fusion, openrouter/pareto-code, openrouter/bodybuilder
```

Explicitly **not** routers: `~anthropic/claude-opus-latest`, `~openai/gpt-*-latest`, and all
other `~…-latest` aliases (real pricing, tokenizer `Router`); all concrete models.

### Detection implementation

Derive the flag once at catalog load and expose a registry helper.

- In `registry.py`'s `ensure_loaded` spec-finalization loop (`registry.py:322-370`), compute
  `is_router` using the existing `_coerce_pricing_number` helper against
  `pricing["prompt"]` / `pricing["completion"]` and `architecture["tokenizer"]`, and store
  `specs[norm_id]["is_router"] = bool(...)`. Missing pricing → `False`.
- Add `OpenRouterModelRegistry.is_router_model(model_id) -> bool` beside `is_zdr_capable`
  (`registry.py:942`). It normalises via `ModelFamily.base_model`, strips a `:variant`
  suffix to the base id (the same base-resolution used by the filters), and returns the spec
  flag. Unknown id → `False`.

This follows the existing pattern: `is_free_model`, `supports_tool_calling`, and
`is_zdr_capable` are module-level/class helpers in `registry.py`, and derived data lives in
the per-model spec.

---

## 4. ZDR opt-in for routers

### New valve

`ZDR_ROUTER_MODELS: str = ""` in `config.py`, in the Models & Catalog area near
`ZDR_MODELS_ONLY` (`config.py:866-889`). Value is a comma-separated list of router ids.

Description (operator-facing, must state the caveat plainly):

> Router-class models expose no ZDR endpoints, so they are hidden by `ZDR_MODELS_ONLY` and
> rejected by `ZDR_ENFORCE` by default. List a router here to opt it in: it becomes
> selectable and, under enforcement, the request is sent with `provider.zdr=true`. This is
> best-effort — the router's downstream model is a separate data path that OpenRouter does
> not guarantee is ZDR.

Parsing normalises each CSV entry with the existing id-normalisation path and compares
against normalised router ids.

### Visibility

In `_apply_model_filters` (`pipe.py:3072-3077`): when `is_router_model(norm_id)` is true,
keep the model iff it is in `ZDR_ROUTER_MODELS`; otherwise skip it. Non-router models keep
the existing endpoint-membership behaviour unchanged.

### Enforcement

In the orchestrator's ZDR gate (`orchestrator.py:680-733`): when the target is a router and
it is in `ZDR_ROUTER_MODELS`, proceed through the existing enforcement path (which sets
`provider.zdr=true`); when it is a router and is **not** listed, reject with a restriction
reason. Non-router models keep the existing `is_zdr_capable` behaviour.

### Explanation

In `_model_restriction_reasons` (`pipe.py:3389-3395`): an unlisted router hidden by
`ZDR_MODELS_ONLY` reports `ZDR_MODELS_ONLY`, so users see the same explanation as any other
ZDR-filtered model. No new user-facing reason string is required.

### Assumption

`openrouter/fusion` is detected as router-class but is managed by the pipe's fusion feature;
this design does not alter the fusion path. If a router is opted into ZDR and happens to be
fusion, the existing fusion endpoint override still applies.

---

## 5. Reasoning for routers

In `_apply_reasoning_preferences` (`reasoning_config.py:49-82`): when
`is_router_model(responses_body.model)`, build the reasoning config from whatever the
request already carries, but **skip the `effort` injection** from `valves.REASONING_EFFORT`.
Still apply `summary` from `REASONING_SUMMARY_MODE` and set `enabled = True`. If the caller
already set an `effort` (directly on the reasoning object, or via the `reasoning_effort`
transform at `transforms.py:312-317`), it is preserved untouched — an explicit user choice
still wins.

In `_apply_task_reasoning_preferences` (`reasoning_config.py:85-112`): skip the task effort
override for routers, for consistency.

The Gemini thinking-config translation and Anthropic verbosity mapping are untouched:
router models never match those family classifiers, so the router's own choice flows through
unmodified. The `reasoning.effort` unsupported-value retry (`orchestrator.py:1168-1236`)
remains as a safety net.

Net effect: by default the pipe asks for reasoning traces but lets the router decide the
effort; users can still force an effort per request.

---

## 6. Pricing classification

- `is_free_model` (`registry.py:1186-1199`): a router with **negative** pricing returns
  `False` explicitly, so a negative sentinel can never be summed into a "free" verdict.
- Zero-priced routers keep normal classification: `openrouter/free` (0/0) still reports as
  free and remains visible under `FREE_MODEL_FILTER=only`. This preserves the free filter's
  purpose.
- Cost display is unchanged. Real cost comes from the response's `usage.cost`; the dashboard
  session tracker already ignores negative catalog rates
  (`session_tracker.py:199-214`, `plugins/pipe_dashboard/plugin.py:32-40`).
- No new valve and no UI change.

---

## 7. Provider routing

Routers have empty `/endpoints` payloads, so provider-routing dropdowns have no providers
for them. When a router appears in `ADMIN_PROVIDER_ROUTING_MODELS` or
`USER_PROVIDER_ROUTING_MODELS`:

- Skip it in `catalog_manager._build_routed_provider_overlay` (`catalog_manager.py:773+`)
  and in `filter_manager.ensure_provider_routing_filters` (`filter_manager.py:2337`), so no
  empty-options filter is generated.
- Log a single warning naming the router and stating that provider routing is skipped.

Non-router models are unaffected.

---

## 8. Observability (minimal)

When `is_router_model(requested)` and the upstream response reports a different `model` id
(seen in the SSE event stream, e.g. `streaming_core.py:1370`, and in the non-streaming
payload at `streaming_core.py:2488-2500`):

- Emit it **once** as a best-effort status line ("Routed via `<id>`") and a debug log.
- Do not alter the Open WebUI `model` field.
- If the field is absent or identical to the requested id, do nothing (silent degradation).

No schema assumptions; this is explicitly best-effort and must not be relied on by other
code.

---

## 9. Data flow summary

```
/models catalog
   └─ registry.ensure_loaded
        └─ spec[norm_id]["is_router"] = pricing sentinel check
             ├─ pipes()._apply_model_filters  → ZDR visibility (routers need ZDR_ROUTER_MODELS)
             ├─ _model_restriction_reasons    → user-facing explanation
             ├─ orchestrator ZDR gate         → allow + provider.zdr=true, or reject
             ├─ reasoning_config              → no effort injection for routers
             ├─ is_free_model                 → negative-priced routers never free
             ├─ provider-routing generation   → skip routers, warn once
             └─ streaming response            → best-effort "Routed via <id>" status
```

---

## 10. Error handling

- Unlisted router under `ZDR_MODELS_ONLY`: dropped from the selector; no error (catalog
  filter).
- Unlisted router under `ZDR_ENFORCE` or a user ZDR request: rejected with the existing ZDR
  restriction reason; if the ZDR list is unavailable, the existing
  `ZDR_ENFORCE_UNAVAILABLE` behaviour applies.
- Missing/renamed upstream `model` field in a response: observability degrades silently.
- Router listed in a provider-routing valve: skipped with a warning; request still proceeds.

---

## 11. Testing

New `tests/test_router_models.py`:

- **Detection:** negative pricing → router; `openrouter/free` (0/0 + tokenizer `Router`) →
  router; `~anthropic/claude-opus-latest` (real pricing + tokenizer `Router`) → not router;
  concrete model → not router; unknown id → not router; `:variant` suffix resolves to base.
- **ZDR visibility:** router hidden with `ZDR_MODELS_ONLY=true` and empty valve; visible when
  listed; non-router behaviour unchanged.
- **ZDR enforcement:** listed router proceeds and the payload carries `provider.zdr=true`;
  unlisted router rejected with the ZDR reason.
- **Reasoning:** router request carries no injected `effort` but does carry `summary` and
  `enabled`; an explicitly supplied effort is preserved; non-router still receives the
  valve effort.
- **Pricing:** negative-priced router is not free; `openrouter/free` is free.
- **Provider routing:** a router in a routing valve is skipped and does not produce an
  empty-options filter.

Update any existing tests that assert the current (pre-change) behaviour on these paths.

---

## 12. Documentation

- New `docs/model_routers.md`: what router-class models are; the 8 currently detected; the
  ZDR caveat and `ZDR_ROUTER_MODELS`; reasoning behaviour; pricing/billing; observability.
- Update `docs/README.md` index.
- Update `docs/model_catalog_and_routing_intelligence.md` (§2 auto-router note).
- Update `docs/openrouter_zdr.md` (router section).
- Update `docs/valves_and_configuration_atlas.md` for `ZDR_ROUTER_MODELS`.
- Add `ZDR_ROUTER_MODELS` to `plugins/pipe_dashboard/config_meta.py` under
  Models & Catalog/ZDR.

No new dependencies; the bundle is unaffected.

---

## 13. Assumptions and open questions

1. **Sentinel stability.** Detection depends on OpenRouter's pricing sentinel (`-1`, or
   `0/0` + tokenizer `Router`). If OpenRouter changes how a router is priced without also
   matching these signals, detection misses it. Accepted; the definition is centralised in
   one helper so it can be updated in one place.
2. **`provider.zdr=true` semantics for routers.** The pipe sends it as it does for any
   opted-in model. Whether OpenRouter propagates it to the router's downstream selection is
   not documented; the design treats it as best-effort and says so in the valve description
   and docs.
3. **Observability scope.** Deliberately minimal (status line + debug log). If OpenRouter
   later publishes a routing-metadata contract, this can be expanded.
4. **No automatic fallback.** Out of scope, per the guide's "no silent fallback" note; users
   retain the opt-in `model_fallback` custom parameter.
