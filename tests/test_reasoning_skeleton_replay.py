"""With tool results not kept, the next turn must still replay each tool round's structure around its reasoning.

A reasoning model that thinks before and after a tool call produces thinking blocks with a tool round between them.
When the round is not kept, replay puts the blocks next to each other, and Anthropic rejects exactly that ("thinking
blocks ... cannot be modified", recorded live in tests/fixtures/anthropic_reasoning_replay_probe.json). So the pipe
keeps a skeleton of each round it ran and did not show as tool cards: the call with empty arguments and a "not
retained" result. The thinking blocks stay apart, while nothing the tool returned, or the model sent it, is stored.

Stage A is the real streaming loop on default tool settings. Stage B feeds the content stage A produced (text plus
hidden markers, and no tool messages, which is what Open WebUI stores when no tool card was shown) to the real
transformer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody, _ToolExecutionContext, generate_item_id
from open_webui_openrouter_pipe.api.transforms import (
    _filter_openrouter_request,
    _responses_payload_to_chat_completions_payload,
)
from open_webui_openrouter_pipe.core.utils import contains_marker
from open_webui_openrouter_pipe.models.reasoning_config import ReasoningConfigManager
from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input

MODEL = "anthropic/claude-opus-4.8"
RESULT_CANARY = "SECRET-TOOL-RESULT-7f3a"
ARGUMENT_CANARY = "SECRET-ARGUMENT-19c2"
FIXTURE = Path(__file__).parent / "fixtures" / "anthropic_reasoning_replay_probe.json"

SEQUENTIAL_TWO_ROUNDS = [("calls", ["toolu-1"]), ("calls", ["toolu-2"]), ("answer", False)]
PARALLEL_WITH_FINAL_REASONING = [("calls", ["toolu-a", "toolu-b"]), ("answer", True)]


def _recorded(group: str, arm: str) -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))[group]["arms"][arm]


def _shape(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("type")) if item.get("type") != "message" else f"message:{item.get('role')}" for item in items]


def _valves(pipe, **changes):
    settings = {
        "TOOL_EXECUTION_MODE": "Pipeline",
        "PERSIST_REASONING_TOKENS": "conversation",
        "PERSIST_TOOL_RESULTS": False,
        "SHOW_TOOL_CARDS": False,
        "MAX_FUNCTION_CALL_LOOPS": 5,
    }
    settings.update(changes)
    return pipe.valves.model_copy(update=settings)


async def _stage_a(pipe, monkeypatch, valves, rounds, *, stream=True, emitter=None, tool_status="completed",
                   real_executor=False, signed=True, message_id: str | None = "m1", real_row_builder=False):
    """Run one turn. Each round is ("calls", [call ids]), which reasons, writes and calls; ("quiet-calls", [call ids]),
    which writes and calls without reasoning; or ("answer", reasons_first). ``signed=False`` streams reasoning with no
    signature, which Anthropic cannot take back. ``message_id=None`` sends no message id, as an API request does;
    ``real_row_builder`` builds rows with the store's own `_make_db_row` instead of a stand-in.

    Returns the content the loop produced, the rows it persisted (ulid -> payload) and the events it emitted.
    """
    step = [0]

    async def model(self, session, request_body, **_kwargs):
        kind, value = rounds[step[0]]
        step[0] += 1
        index = step[0]
        output: list[dict[str, Any]] = []
        if kind == "calls" or (kind == "answer" and value):
            block = {"type": "reasoning", "id": f"rs-{index}", "status": "completed",
                     "content": [{"type": "reasoning_text", "text": f"THOUGHT-{index}"}], "summary": []}
            if signed:
                block["signature"] = f"sig-{index}"
            yield {"type": "response.output_item.done", "item": block}
            output.append(block)
        yield {"type": "response.output_text.delta", "delta": f"text {index} "}
        if kind in ("calls", "quiet-calls"):
            for call_id in value:
                call = {"type": "function_call", "call_id": call_id, "name": "lookup",
                        "arguments": json.dumps({"q": ARGUMENT_CANARY}), "status": "completed"}
                yield {"type": "response.output_item.done", "item": call}
                output.append(call)
        yield {"type": "response.completed", "response": {"output": output, "usage": {}}}

    async def lookup(**_kwargs):
        return RESULT_CANARY

    async def run_tools(calls, _registry):
        return [{"type": "function_call_output", "call_id": c.get("call_id"), "output": RESULT_CANARY,
                 "status": tool_status} for c in calls]

    persisted: dict[str, dict[str, Any]] = {}

    def make_row(_chat_id, _message_id, _model_id, payload):
        return {"payload": payload}

    async def persist(rows):
        ulids = [generate_item_id() for _ in rows]
        persisted.update(zip(ulids, (row["payload"] for row in rows)))
        return ulids

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model)
    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", model)
    if not real_executor:
        monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", run_tools)
    if real_row_builder:
        monkeypatch.setattr(pipe._artifact_store, "_item_model", object())
    else:
        monkeypatch.setattr(pipe._artifact_store, "_make_db_row", make_row)
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", persist)

    emitted: list[dict[str, Any]] = []

    async def capture(event):
        emitted.append(event)

    body = ResponsesBody(model=MODEL, input=[], stream=stream)
    registry = {"lookup": {"type": "function", "callable": lookup,
                           "spec": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}}
    handler = pipe._streaming_handler
    context = token = None
    if real_executor:
        context = _ToolExecutionContext(queue=asyncio.Queue(maxsize=50), per_request_semaphore=asyncio.Semaphore(1),
                                        global_semaphore=None, timeout=5.0, batch_timeout=5.0, idle_timeout=None,
                                        user_id="u1", event_emitter=None, batch_cap=1)
        context.workers.append(asyncio.create_task(pipe._ensure_tool_executor()._tool_worker_loop(context)))
        token = pipe._TOOL_CONTEXT.set(context)
    try:
        runner = handler._run_streaming_loop if stream else handler._run_nonstreaming_loop
        content = await runner(
            body, valves, capture if emitter is None else emitter,
            metadata={"model": {"id": MODEL}, "chat_id": "c1", **({"message_id": message_id} if message_id else {})},
            tools=registry, session=cast(Any, object()), user_id="u1",
        )
    finally:
        if context is not None:
            pipe._TOOL_CONTEXT.reset(token)
            for worker in context.workers:
                worker.cancel()
            await asyncio.gather(*context.workers, return_exceptions=True)
    return content, persisted, emitted


async def _stage_b(pipe, valves, content, persisted, *, refs=None):
    async def loader(_chat_id, _message_id, ulids):
        return {u: persisted[u] for u in ulids if u in persisted}

    return await transform_messages_to_input(
        pipe,
        [{"role": "user", "content": "q1"},
         {"role": "assistant", "message_id": "m1", "content": content},
         {"role": "user", "content": "q2"}],
        chat_id="c1", openwebui_model_id="owui", artifact_loader=loader, model_id=MODEL, valves=valves,
        replayed_reasoning_refs=refs,
    )


def _has_consecutive_reasoning(items):
    positions = [i for i, item in enumerate(items) if item.get("type") == "reasoning"]
    return any(b == a + 1 for a, b in zip(positions, positions[1:]))


# --- the replay is a shape Anthropic accepted --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rounds", "group", "arm"),
    [
        (SEQUENTIAL_TWO_ROUNDS, "sequential_rounds", "V1_STUB"),
        (PARALLEL_WITH_FINAL_REASONING, "parallel_calls_with_final_round_reasoning", "PIPE_STUB"),
    ],
    ids=["two-sequential-rounds", "parallel-calls-then-reasoning"],
)
async def test_an_unretained_tool_turn_replays_as_a_shape_anthropic_accepted(
    monkeypatch, pipe_instance_async, rounds, group, arm
):
    pipe = pipe_instance_async
    valves = _valves(pipe)
    accepted = _recorded(group, arm)
    assert accepted["status"] == 200

    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, rounds)
    replay = await _stage_b(pipe, valves, content, persisted)

    assert _shape(replay) == accepted["shape"]
    assert _shape(replay) != _recorded("sequential_rounds", "T52")["shape"]


# --- every default-settings path keeps the rounds apart ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", [1, 2])
@pytest.mark.parametrize(
    ("stream", "cards"), [(True, False), (False, True), (False, False)],
    ids=["streaming-cards-off", "not-streaming-cards-on", "not-streaming-cards-off"],
)
async def test_every_executed_round_keeps_its_structure_when_no_card_holds_it(
    monkeypatch, pipe_instance_async, rounds, stream, cards
):
    # Cards are only shown while streaming, so a non-streaming turn with the cards setting on still needs the skeleton.
    pipe = pipe_instance_async
    valves = _valves(pipe, SHOW_TOOL_CARDS=cards)
    turn = [("calls", [f"toolu-{i}"]) for i in range(rounds)] + [("answer", True)]

    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, turn, stream=stream)
    replay = await _stage_b(pipe, valves, content, persisted)

    assert not _has_consecutive_reasoning(replay), _shape(replay)
    calls = [item.get("call_id") for item in replay if item.get("type") == "function_call"]
    results = [item.get("call_id") for item in replay if item.get("type") == "function_call_output"]
    assert calls == results == [f"toolu-{i}" for i in range(rounds)]


@pytest.mark.asyncio
async def test_a_round_that_did_not_reason_keeps_its_skeleton_so_later_reasoning_finds_its_call(
    monkeypatch, pipe_instance_async
):
    """Reasoning is placed by the ordinal of the call beside it, counted over every call in the turn, so a round that
    produced no reasoning of its own still has to leave its call in the replay."""
    pipe = pipe_instance_async
    valves = _valves(pipe)
    turn = [("quiet-calls", ["toolu-1"]), ("calls", ["toolu-2"]), ("answer", True)]

    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, turn)
    replay = await _stage_b(pipe, valves, content, persisted)

    assert _shape(replay) == [
        "message:user", "message:assistant", "function_call", "function_call_output",
        "reasoning", "function_call", "function_call_output", "reasoning", "message:user",
    ]


# --- nothing the tool returned or the model sent it is stored -----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "not-streaming"])
async def test_the_skeleton_keeps_neither_the_tool_result_nor_the_arguments(monkeypatch, pipe_instance_async, stream):
    pipe = pipe_instance_async
    valves = _valves(pipe)

    _, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS, stream=stream)

    stored = json.dumps(persisted)
    assert [p.get("type") for p in persisted.values()].count("function_call") == 2
    assert RESULT_CANARY not in stored
    assert ARGUMENT_CANARY not in stored


# --- the skeleton lives and dies with its reasoning --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_skeleton_is_kept_when_reasoning_is_not_kept(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    valves = _valves(pipe, PERSIST_REASONING_TOKENS="disabled")

    _, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS)

    assert [p.get("type") for p in persisted.values()] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "not-streaming"])
async def test_a_tool_turn_that_never_reasoned_keeps_no_skeleton(monkeypatch, pipe_instance_async, stream):
    # The skeleton exists only to keep reasoning apart; a turn with no reasoning leaves no rows and no markers.
    pipe = pipe_instance_async
    valves = _valves(pipe)
    turn = [("quiet-calls", ["toolu-1"]), ("quiet-calls", ["toolu-2"]), ("answer", False)]

    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, turn, stream=stream)

    assert persisted == {}
    assert not contains_marker(content)


@pytest.mark.asyncio
async def test_the_skeleton_is_cleaned_up_with_the_reasoning_after_its_next_reply(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    valves = _valves(pipe, PERSIST_REASONING_TOKENS="next_reply")
    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS)
    assert [p.get("type") for p in persisted.values()].count("function_call") == 2, persisted

    refs: list[tuple[str, str]] = []
    await _stage_b(pipe, valves, content, persisted, refs=refs)

    assert sorted(ulid for _, ulid in refs) == sorted(persisted)


@pytest.mark.asyncio
async def test_a_skeleton_whose_reasoning_is_gone_is_not_replayed(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    valves = _valves(pipe)
    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS)
    without_reasoning = {u: p for u, p in persisted.items() if p.get("type") != "reasoning"}
    assert [p.get("type") for p in without_reasoning.values()].count("function_call") == 2, persisted

    replay = await _stage_b(pipe, valves, content, without_reasoning)

    assert _shape(replay) == ["message:user", "message:assistant", "message:user"]


# --- the skeleton only stands in where nothing else holds the round -----------------------------------------------------


@pytest.mark.asyncio
async def test_rounds_shown_as_tool_cards_get_no_skeleton(monkeypatch, pipe_instance_async):
    # With both cards shown, Open WebUI stores the call and its result itself; a skeleton would replay the call twice.
    pipe = pipe_instance_async
    valves = _valves(pipe, SHOW_TOOL_CARDS=True)

    _, persisted, emitted = await _stage_a(
        pipe, monkeypatch, valves, [("calls", ["toolu-1"]), ("answer", True)], stream=True, real_executor=True,
    )

    carded = [
        event.get("item", {}).get("type")
        for event in emitted
        if event.get("type") == "response.output_item.added"
    ]
    assert "function_call" in carded and "function_call_output" in carded, carded
    assert [p.get("type") for p in persisted.values() if p.get("type") != "reasoning"] == []


@pytest.mark.asyncio
async def test_the_skeleton_is_never_published_as_turn_output(monkeypatch, pipe_instance_async):
    # A published call without its result is one Open WebUI would run again.
    pipe = pipe_instance_async
    valves = _valves(pipe)

    _, persisted, emitted = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS, stream=True)
    assert [p.get("type") for p in persisted.values()].count("function_call") == 2, persisted

    published = [
        item
        for event in emitted if event.get("type") == "response.completed"
        for item in (event.get("response") or {}).get("output") or []
    ]
    assert [item.get("type") for item in published if item.get("type") in ("function_call", "function_call_output")] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_status", ["completed", "incomplete"])
async def test_a_skeleton_result_keeps_the_real_status(monkeypatch, pipe_instance_async, tool_status):
    pipe = pipe_instance_async
    valves = _valves(pipe)

    _, persisted, _ = await _stage_a(
        pipe, monkeypatch, valves, [("calls", ["toolu-1"]), ("answer", True)], tool_status=tool_status,
    )

    results = [p for p in persisted.values() if p.get("type") == "function_call_output"]
    assert [r.get("status") for r in results] == [tool_status]



@pytest.mark.asyncio
async def test_a_round_cut_by_the_loop_limit_keeps_its_skeleton_though_its_result_card_was_shown(
    monkeypatch, pipe_instance_async
):
    """At the loop limit the pipe answers the call with a stub instead of running it, and shows only the stub's result
    card. Open WebUI drops a result whose call it never received when it rebuilds the history, so only the skeleton
    keeps that round between its reasoning blocks."""
    pipe = pipe_instance_async
    valves = _valves(pipe, SHOW_TOOL_CARDS=True, MAX_FUNCTION_CALL_LOOPS=1)

    content, persisted, emitted = await _stage_a(
        pipe, monkeypatch, valves, [("calls", ["toolu-1"]), ("answer", True)], stream=True
    )
    carded = [
        event.get("item", {}).get("type")
        for event in emitted
        if event.get("type") == "response.output_item.added"
    ]
    assert "function_call_output" in carded and "function_call" not in carded, carded

    replay = await _stage_b(pipe, valves, content, persisted)

    assert _shape(replay) == [
        "message:user", "message:assistant", "reasoning", "function_call", "function_call_output", "reasoning",
        "message:user",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("retention", "warns"), [("disabled", False), ("conversation", True)])
async def test_a_request_without_a_message_id_warns_only_when_it_had_something_to_keep(
    monkeypatch, pipe_instance_async, caplog, retention, warns
):
    """A request without a message id, as an API call sends it, cannot store anything. With reasoning not kept it had
    nothing to store, so a warning that its artifacts were skipped would be false."""
    pipe = pipe_instance_async
    valves = _valves(pipe, PERSIST_REASONING_TOKENS=retention)
    caplog.set_level(logging.WARNING)

    content, persisted, _ = await _stage_a(
        pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS, message_id=None, real_row_builder=True
    )

    assert "text 3" in content
    assert persisted == {}
    assert any("missing message_id" in record.getMessage() for record in caplog.records) is warns


# --- every stage that drops reasoning drops the skeleton with it -------------------------------------------------------


async def _outgoing_body(pipe, valves, content, persisted):
    """The next turn's request as the streaming loop holds it: the real replay, then the real sanitizer."""
    body = ResponsesBody(model=MODEL, input=await _stage_b(pipe, valves, content, persisted), stream=True)
    _sanitize_request_input(pipe, body)
    return body


def _internal_keys(items: list[Any]) -> list[str]:
    return [str(key) for item in items if isinstance(item, dict) for key in item if str(key).startswith("_anchor")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rounds", "group", "arm"),
    [
        (SEQUENTIAL_TWO_ROUNDS, "sequential_rounds", "V1_STUB"),
        (PARALLEL_WITH_FINAL_REASONING, "parallel_calls_with_final_round_reasoning", "PIPE_STUB"),
    ],
    ids=["two-sequential-rounds", "parallel-calls-then-reasoning"],
)
async def test_the_responses_request_on_the_wire_is_the_accepted_shape_with_no_internal_keys(
    monkeypatch, pipe_instance_async, rounds, group, arm
):
    pipe = pipe_instance_async
    valves = _valves(pipe)
    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, rounds)
    body = await _outgoing_body(pipe, valves, content, persisted)

    wire = _filter_openrouter_request(body.model_dump(exclude_none=True))

    assert _shape(wire["input"]) == _recorded(group, arm)["shape"]
    assert _internal_keys(wire["input"]) == []


@pytest.mark.asyncio
async def test_the_chat_completions_fallback_drops_the_skeleton_with_the_reasoning_it_cannot_carry(
    monkeypatch, pipe_instance_async
):
    # The fallback converts the same request dict the /responses attempt was filtered from, so that filtering must
    # leave the dict able to tell a skeleton round from a real one.
    pipe = pipe_instance_async
    valves = _valves(pipe)
    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS)
    request = (await _outgoing_body(pipe, valves, content, persisted)).model_dump(exclude_none=True)
    _filter_openrouter_request(request)

    chat = _responses_payload_to_chat_completions_payload(request)["messages"]

    assert [message.get("role") for message in chat] == ["user", "assistant", "user"], chat
    assert _internal_keys(chat) == []


@pytest.mark.asyncio
async def test_the_thinking_signature_retry_drops_the_skeleton_with_the_reasoning(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    valves = _valves(pipe)
    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS)
    body = await _outgoing_body(pipe, valves, content, persisted)
    assert isinstance(body.input, list)
    assert "reasoning" in _shape(body.input), _shape(body.input)

    assert ReasoningConfigManager._strip_replayed_reasoning(body) is True

    assert _shape(body.input) == ["message:user", "message:assistant", "message:user"]


@pytest.mark.asyncio
async def test_unsigned_reasoning_removed_for_anthropic_takes_its_skeleton_with_it(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    valves = _valves(pipe)
    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS, signed=False)
    kinds = [p.get("type") for p in persisted.values()]
    assert kinds.count("reasoning") == 2 and kinds.count("function_call") == 2, kinds

    body = await _outgoing_body(pipe, valves, content, persisted)

    assert isinstance(body.input, list)
    assert _shape(body.input) == ["message:user", "message:assistant", "message:user"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["streaming-forced-chat", "streaming-fallback", "non-streaming-forced-chat", "non-streaming-fallback"]
)
async def test_every_chat_completions_route_hands_on_a_request_that_can_still_drop_the_skeleton(
    monkeypatch, pipe_instance_async, route
):
    """The gateway passes the chat adapter the request the loop built; converting it must still tell a skeleton round
    from a real one, so the skeleton leaves with the reasoning the chat format cannot carry."""
    pipe = pipe_instance_async
    valves = _valves(pipe, AUTO_FALLBACK_CHAT_COMPLETIONS=True)
    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS)
    request = (await _outgoing_body(pipe, valves, content, persisted)).model_dump(exclude_none=True)
    monkeypatch.undo()
    received: list[dict[str, Any]] = []

    async def chat_streaming(self, session, responses_request_body, **_kwargs):
        received.append(responses_request_body)
        yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    async def chat_nonstreaming(self, session, responses_request_body, **_kwargs):
        received.append(responses_request_body)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}

    unsupported = RuntimeError("this model does not support the responses endpoint")

    async def responses_streaming(self, *_args, **_kwargs):
        raise unsupported
        yield {}

    async def responses_nonstreaming(self, *_args, **_kwargs):
        raise unsupported

    monkeypatch.setattr(Pipe, "send_openai_chat_completions_streaming_request", chat_streaming)
    monkeypatch.setattr(Pipe, "send_openai_chat_completions_nonstreaming_request", chat_nonstreaming)
    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", responses_streaming)
    monkeypatch.setattr(Pipe, "send_openai_responses_nonstreaming_request", responses_nonstreaming)

    session = cast(Any, object())
    override = "chat_completions" if route.endswith("forced-chat") else None
    gateway = (
        pipe.send_openrouter_streaming_request if route.startswith("streaming")
        else pipe.send_openrouter_nonstreaming_request_as_events
    )
    async for _ in gateway(session, request, "sk-test", "https://openrouter.ai/api/v1", valves=valves,
                          endpoint_override=override):
        pass

    assert len(received) == 1, route
    chat = _responses_payload_to_chat_completions_payload(received[0])["messages"]
    assert [message.get("role") for message in chat] == ["user", "assistant", "user"], chat
    assert _internal_keys(chat) == []


@pytest.mark.asyncio
async def test_removing_one_turns_unsigned_reasoning_takes_only_that_turns_skeleton(monkeypatch, pipe_instance_async):
    """Unsigned reasoning is removed turn by turn, so a later signed turn keeps its reasoning and its skeleton rounds."""
    pipe = pipe_instance_async
    valves = _valves(pipe)
    first, first_rows, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS, signed=False)
    second, second_rows, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS, signed=True)
    persisted = {**first_rows, **second_rows}

    async def loader(_chat_id, _message_id, ulids):
        return {u: persisted[u] for u in ulids if u in persisted}

    replay = await transform_messages_to_input(
        pipe,
        [{"role": "user", "content": "q1"}, {"role": "assistant", "message_id": "m1", "content": first},
         {"role": "user", "content": "q2"}, {"role": "assistant", "message_id": "m2", "content": second},
         {"role": "user", "content": "q3"}],
        chat_id="c1", openwebui_model_id="owui", artifact_loader=loader, model_id=MODEL, valves=valves,
    )
    body = ResponsesBody(model=MODEL, input=replay, stream=True)
    _sanitize_request_input(pipe, body)

    assert isinstance(body.input, list)
    assert _shape(body.input) == [
        "message:user", "message:assistant", *_recorded("sequential_rounds", "V1_STUB")["shape"]
    ]


# --- a Continue of an unretained tool turn -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_continue_of_an_unretained_tool_turn_sends_the_shape_anthropic_accepted(monkeypatch, pipe_instance_async):
    """A Continue sends the history up to the continued message, so its request ends inside that turn. Ending on the
    turn's skeleton result was accepted; dropping those rounds leaves the request ending on the assistant's own text,
    which the model rejected ("does not support assistant message prefill")."""
    pipe = pipe_instance_async
    valves = _valves(pipe)
    accepted = _recorded("continue_of_a_tool_turn", "A_FULL")
    rejected = _recorded("continue_of_a_tool_turn", "B_FULL")
    assert (accepted["status"], rejected["status"]) == (200, 400)

    content, persisted, _ = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS)

    async def loader(_chat_id, _message_id, ulids):
        return {u: persisted[u] for u in ulids if u in persisted}

    continue_input = await transform_messages_to_input(
        pipe,
        [{"role": "user", "content": "q1"}, {"role": "assistant", "message_id": "m1", "content": content}],
        chat_id="c1", openwebui_model_id="owui", artifact_loader=loader, model_id=MODEL, valves=valves,
    )
    body = ResponsesBody(model=MODEL, input=continue_input, stream=True)
    _sanitize_request_input(pipe, body)
    wire = _filter_openrouter_request(body.model_dump(exclude_none=True))

    assert _shape(wire["input"]) == accepted["shape"]
    assert _shape(wire["input"]) != rejected["shape"]
