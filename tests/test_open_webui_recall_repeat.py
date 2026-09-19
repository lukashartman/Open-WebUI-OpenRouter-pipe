"""A continued answer whose tools Open WebUI runs reaches OpenRouter with each earlier item once.

In Open-WebUI tool mode Open WebUI runs the tools itself and calls the pipe again with
`[*messages, *convert_output_to_messages(output)]`. On a Continue `messages` already ends with the continued message,
and `output` comes to hold that message's stored items again: from the re-call, which sets them aside at the start and
fold them back into `output` after each re-call. So Open WebUI sends the continued message twice.

Open WebUI's own `convert_output_to_messages`, `process_messages_with_output` and `handle_responses_streaming_event`
are compiled from the installed Open WebUI; the two folding rules follow each version's streaming handler.
"""

from __future__ import annotations

import __future__
import ast
import copy
import json
import sysconfig
from pathlib import Path
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe.pipe as pipe_mod
import open_webui_openrouter_pipe.streaming.streaming_core as streaming_core_mod
from open_webui_openrouter_pipe import Pipe, ResponsesBody, generate_item_id
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input

MODEL = "anthropic/claude-opus-4.8"
LABELS = ("R1", "R2", "R3", "R4")


def _open_webui_functions() -> dict[str, Any]:
    utils = Path(sysconfig.get_paths()["purelib"]) / "open_webui" / "utils"
    wanted = {
        "misc.py": {"convert_output_to_messages", "reconcile_tool_pairs", "get_content_from_message", "get_output_text"},
        "middleware.py": {"handle_responses_streaming_event", "process_messages_with_output", "deep_merge"},
    }
    namespace: dict[str, Any] = {"json": json}
    for filename, names in wanted.items():
        source = utils / filename
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == names
        code = compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec",
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)
        exec(code, namespace)
    return namespace


class _StoredMessage:
    """The output Open WebUI's streaming handler holds for one assistant message while it runs the message's tools."""

    def __init__(self, open_webui: dict[str, Any], *, stored: list[dict[str, Any]], continued: bool):
        self._open_webui = open_webui
        stored = copy.deepcopy(stored)
        self.prior_output, self.output = (stored, []) if stored and continued else ([], stored)
        self._tool_calls: list[dict[str, Any]] = []
        self._count = 0

    def full_output(self) -> list[dict[str, Any]]:
        return self.prior_output + self.output

    async def feed(self, event: dict[str, Any]) -> None:
        kind = event.get("type") or ""
        if kind.startswith("response."):
            self.output, _ = self._open_webui["handle_responses_streaming_event"](copy.deepcopy(event), self.output)
        elif kind == "chat:message:delta":
            text = (event.get("data") or {}).get("content") or ""
            if text:
                if not self.output or self.output[-1].get("type") != "message":
                    self._count += 1
                    self.output.append({"type": "message", "id": f"msg-owui-{self._count}", "status": "in_progress",
                                        "role": "assistant", "content": [{"type": "output_text", "text": ""}]})
                self.output[-1]["content"][-1]["text"] += text
        elif kind == "chat:tool_calls":
            for delta in (event.get("data") or {}).get("tool_calls") or []:
                function = delta.get("function") or {}
                current = next((call for call in self._tool_calls if call["index"] == delta.get("index")), None)
                if current is None:
                    self._tool_calls.append({"index": delta.get("index"), "id": delta.get("id"),
                                             "name": function.get("name") or "", "arguments": function.get("arguments") or ""})
                else:
                    current["name"] = function.get("name") or current["name"]
                    current["arguments"] += function.get("arguments") or ""

    def run_tools(self) -> None:
        calls, self._tool_calls = self._tool_calls, []
        known = {item.get("call_id") for item in self.output if item.get("type") == "function_call"}
        for call in calls:
            if call["id"] not in known:
                self.output.append({"type": "function_call", "id": call["id"], "call_id": call["id"],
                                    "name": call["name"], "arguments": call["arguments"] or "{}", "status": "in_progress"})
        for call in calls:
            self._count += 1
            self.output.append({"type": "function_call_output", "id": f"fco-owui-{self._count}", "call_id": call["id"],
                                "output": [{"type": "input_text", "text": "ok"}], "status": "completed"})
        for item in self.output:
            if item.get("type") == "function_call":
                item["status"] = "completed"

    def begin_recall(self) -> None:
        self.prior_output = self.full_output()
        self.output = []

    def end_recall(self) -> None:
        self.output[:0] = self.prior_output
        self.prior_output = []


def _reasoning(label: str) -> dict[str, Any]:
    return {"type": "reasoning", "id": f"rs-{label}", "status": "completed",
            "content": [{"type": "reasoning_text", "text": label}], "summary": [], "signature": f"sig-{label}"}


def _model_round(steps: list[tuple[str, str]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    listed: list[dict[str, Any]] = []
    for kind, value in steps:
        if kind == "reason":
            item = _reasoning(value)
        elif kind == "call":
            item = {"type": "function_call", "call_id": value, "name": "lookup", "arguments": "{}", "status": "completed"}
        else:
            events.append({"type": "response.output_text.delta", "delta": value})
            listed.append({"type": "message", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": value}]})
            continue
        events.append({"type": "response.output_item.done", "item": item})
        listed.append(item)
    events.append({"type": "response.completed", "response": {"output": listed, "usage": {}}})
    return events


def _label(item: dict[str, Any]) -> str:
    kind = item.get("type")
    if kind == "reasoning":
        return next((label for label in LABELS if label in json.dumps(item)), "reasoning")
    if kind == "function_call":
        return str(item.get("call_id"))
    if kind == "function_call_output":
        return f"result:{item.get('call_id')}"
    if kind == "message":
        parts = item.get("content")
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)) if isinstance(parts, list) else str(parts)
        return f"{item.get('role')}:{text.strip()}"
    return str(kind)


async def _answer(pipe, monkeypatch, open_webui, persisted, *, rounds, stored, continued, sent, earlier):
    """One answer: the pipe's first request, then Open WebUI's re-call after each round whose tools it ran."""

    async def loader(_chat_id, _message_id, ulids):
        return {ulid: persisted[ulid] for ulid in ulids if ulid in persisted}

    def make_row(_chat_id, _message_id, _model_id, payload):
        return {"payload": payload}

    async def persist(rows):
        ulids = [generate_item_id() for _ in rows]
        persisted.update(zip(ulids, (row["payload"] for row in rows)))
        return ulids

    class _Chats:
        @staticmethod
        async def get_message_by_id_and_message_id(_chat_id, _message_id):
            return {"output": copy.deepcopy(stored)}

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", make_row)
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", persist)
    monkeypatch.setattr(pipe._artifact_store, "_db_fetch", loader)
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test", None))
    monkeypatch.setattr(streaming_core_mod, "Chats", _Chats)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry, "list_models",
        lambda: [{"id": MODEL, "name": "Claude Opus 4.8", "norm_id": "anthropic.claude-opus-4.8"}],
    )
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Open-WebUI", "PERSIST_REASONING_TOKENS": "conversation"})
    message = _StoredMessage(open_webui, stored=stored, continued=continued)
    if continued:
        opening = open_webui["process_messages_with_output"](
            [*earlier, {"role": "user", "content": "q1"}, {"role": "assistant", "content": "", "output": stored}],
            reasoning_format=None,
        )
    else:
        opening = [*earlier, {"role": "user", "content": "q1"}]
    metadata = {"model": {"id": MODEL}, "chat_id": "c1", "message_id": "m1"}
    if continued:
        metadata["assistant_message_id"] = "m1"
    for index, steps in enumerate(rounds):
        if index == 0:
            messages = list(opening)
        else:
            messages = opening + open_webui["convert_output_to_messages"](
                message.output, raw=True, reasoning_format=None, flatten_tool_images=True
            )
            message.begin_recall()

        async def model(self, session, request_body, _events=_model_round(steps), **_kwargs):
            sent.append(copy.deepcopy(request_body))
            for event in _events:
                yield event

        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model)
        await pipe._handle_pipe_call(
            {"stream": True, "model": MODEL, "messages": messages}, {"id": "u1"}, None, message.feed, None,
            dict(metadata), {}, None, None, valves=valves, session=cast(Any, object()),
        )
        if index > 0:
            message.end_recall()
        message.run_tools()
    return message.full_output()


def _repeated(request_body: dict[str, Any]) -> list[str]:
    keys = [(item.get("type"), item.get("id") if item.get("type") == "reasoning" else item.get("call_id"))
            for item in request_body.get("input") or [] if item.get("type") != "message"]
    return sorted({f"{kind}:{key}" for kind, key in keys if keys.count((kind, key)) > 1})


EARLIER_TURNS = {
    "no-earlier-turn": [],
    "an-earlier-turn": [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "Earlier answer."}],
}

FIRST_ANSWERS = {
    "answered": [[("reason", "R1"), ("text", "Part one.")]],
    "used-a-tool": [[("reason", "R1"), ("call", "c0")], [("text", "Part one.")]],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier", list(EARLIER_TURNS))
@pytest.mark.parametrize("first_answer", list(FIRST_ANSWERS))
@pytest.mark.parametrize("tool_rounds", [1, 2], ids=["one-tool-round", "two-tool-rounds"])
async def test_a_continued_answer_whose_tools_open_webui_runs_sends_each_earlier_item_upstream_once(
    monkeypatch, pipe_instance_async, tool_rounds, first_answer, earlier
):
    pipe = pipe_instance_async
    open_webui = _open_webui_functions()
    persisted: dict[str, dict[str, Any]] = {}
    first_sent: list[dict[str, Any]] = []
    stored = await _answer(
        pipe, monkeypatch, open_webui, persisted, rounds=FIRST_ANSWERS[first_answer], stored=[],
        continued=False, sent=first_sent, earlier=EARLIER_TURNS[earlier],
    )
    continuation = [[("reason", "R2"), ("call", "c1")]]
    if tool_rounds == 2:
        continuation.append([("reason", "R3"), ("call", "c2")])
    continuation.append([("reason", "R4"), ("text", "Part two.")])
    sent: list[dict[str, Any]] = []
    stored_after = await _answer(
        pipe, monkeypatch, open_webui, persisted, rounds=continuation, stored=stored, continued=True,
        sent=sent, earlier=EARLIER_TURNS[earlier],
    )

    assert len(sent) == len(continuation)
    assert [_repeated(body) for body in sent] == [[] for _ in sent], [[_label(i) for i in body["input"]] for body in sent]

    async def loader(_chat_id, _message_id, ulids):
        return {ulid: persisted[ulid] for ulid in ulids if ulid in persisted}

    next_turn = open_webui["process_messages_with_output"](
        [*EARLIER_TURNS[earlier], {"role": "user", "content": "q1"}, {"role": "assistant", "content": "", "output": stored_after},
         {"role": "user", "content": "q2"}],
        reasoning_format=None,
    )
    replay = await transform_messages_to_input(
        pipe, next_turn, chat_id="c1", openwebui_model_id="owui", artifact_loader=loader, model_id=MODEL,
        valves=pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Open-WebUI", "PERSIST_REASONING_TOKENS": "conversation"}),
    )
    first_part = ["R1", "c0", "result:c0", "assistant:Part one."] if first_answer == "used-a-tool" else ["R1", "assistant:Part one."]
    tool_rounds_part = ["R2", "c1", "result:c1", "R3", "c2", "result:c2"][: 3 * tool_rounds]
    earlier_part = ["user:q0", "assistant:Earlier answer."] if EARLIER_TURNS[earlier] else []
    assert [_label(item) for item in replay] == [
        *earlier_part, "user:q1", *first_part, *tool_rounds_part, "R4", "assistant:Part two.", "user:q2"
    ]


def _owui_round(call_id: str) -> list[dict[str, Any]]:
    """One tool round in the shape Open WebUI's own converter hands the pipe."""
    return [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "ok"},
    ]


@pytest.mark.parametrize(
    "call_ids", [("c0", "c1"), ("x0", "x0")], ids=["distinct-call-ids", "two-rounds-sharing-a-call-id"]
)
@pytest.mark.parametrize(
    "resent", [False, True], ids=["the-first-request-of-a-continue", "an-open-webui-re-call"]
)
def test_only_a_turn_open_webui_really_re_sent_is_taken_out_of_the_request(call_ids, resent):
    """The collapse removes a block only when Open WebUI sent that block twice.

    Open WebUI appends its in-memory copy of the turn on a re-call, so the request carries the turn
    twice and the pipe drops the first copy. On the first request of a Continue nothing is repeated,
    and a message whose own rounds happen to share a `call_id` - which both chat-completions adapters
    produce, since they number ids per request - must not be read as such a repeat.

    The four arms differ only in whether the turn was re-sent and whether its ids repeat, and each
    asserts the whole list, because "shorter than it was" and "'Part one.' appears once" are both true
    of a request that has silently lost a round.
    """
    from open_webui_openrouter_pipe.requests.orchestrator import _without_resent_continued_turn

    turn = [
        {"role": "user", "content": "q1"},
        *_owui_round(call_ids[0]),
        *_owui_round(call_ids[1]),
        {"role": "assistant", "content": "Part one."},
    ]
    if not resent:
        assert _without_resent_continued_turn(list(turn)) == turn
        return

    # Open WebUI appends its whole in-memory copy of the turn, which on a re-call ends on the round it
    # has just run, so the request ends on a tool result rather than on finished text.
    re_called = [*turn[1:], *_owui_round("c9")]
    assert _without_resent_continued_turn(turn + re_called) == [turn[0], *re_called]


def test_two_unrelated_answers_in_one_turn_are_both_kept():
    """A continuation extends the text it continues; two different answers are not a re-sent turn.

    Open WebUI with realtime chat save writes a partial answer mid-stream, so a continued
    message can hold two assistant messages whose texts are unrelated. Dropping the prefix test would
    make the collapse delete the first of any two adjacent assistant messages.
    """
    from open_webui_openrouter_pipe.requests.orchestrator import _without_resent_continued_turn

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Part one."},
        {"role": "assistant", "content": "A completely different answer."},
        *_owui_round("c0"),
    ]
    assert _without_resent_continued_turn(list(messages)) == messages


def test_a_model_repeating_itself_at_the_end_of_a_turn_keeps_both_rounds():
    """A block Open WebUI re-sent is never the last thing in the request.

    Open WebUI re-calls the pipe *because* it has just run a tool, and it appends that round after its
    copy of the turn, so something always follows the repeat. A repeat that ends the request is the
    model itself calling the same tool twice - ordinary, because both chat-completions adapters number
    call ids per request, so two rounds of one turn share an id, and a retried call has the same name,
    arguments and result text as well.

    This is the shape a Continue sends when the generation was stopped inside a tool round: the stored
    output ends on a tool result, so the request does too.
    """
    from open_webui_openrouter_pipe.requests.orchestrator import _without_resent_continued_turn

    messages = [
        {"role": "user", "content": "q1"},
        *_owui_round("toolcall-m1-0"),
        *_owui_round("toolcall-m1-0"),
    ]

    assert _without_resent_continued_turn(list(messages)) == messages
