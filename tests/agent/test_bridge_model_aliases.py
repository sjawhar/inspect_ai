"""Tests for model_aliases in resolve_inspect_model."""

import pytest

from inspect_ai.agent._bridge.util import resolve_inspect_model
from inspect_ai.model._model import Model, get_model


def test_resolve_inspect_model_bare_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_EVAL_MODEL", "mockllm/default")
    model = resolve_inspect_model("inspect")
    assert str(model) == "mockllm/default"


def test_resolve_inspect_model_prefixed() -> None:
    model = resolve_inspect_model("inspect/mockllm/model")
    assert str(model) == "mockllm/model"


def test_resolve_inspect_model_alias_takes_priority() -> None:
    target = get_model("mockllm/alias-target")
    aliases: dict[str, str | Model] = {"my-alias": target}
    result = resolve_inspect_model("my-alias", model_aliases=aliases)
    assert result is target


def test_resolve_inspect_model_alias_string() -> None:
    aliases: dict[str, str | Model] = {"my-alias": "mockllm/alias-target"}
    result = resolve_inspect_model("my-alias", model_aliases=aliases)
    assert str(result) == "mockllm/alias-target"


def test_resolve_inspect_model_fallback_used_for_non_inspect() -> None:
    result = resolve_inspect_model(
        "some-random-model", fallback_model="inspect/mockllm/fallback"
    )
    assert str(result) == "mockllm/fallback"


def test_resolve_inspect_model_alias_over_fallback() -> None:
    """Aliases are checked before fallback."""
    target = get_model("mockllm/alias-target")
    aliases: dict[str, str | Model] = {"my-name": target}
    result = resolve_inspect_model(
        "my-name", model_aliases=aliases, fallback_model="inspect/mockllm/other"
    )
    assert result is target


class _CollectingSink:
    """ModelEventSink that keeps every event handed to it."""

    def __init__(self) -> None:
        self.pending: list[object] = []
        self.complete: list[object] = []

    def on_pending(self, event: object) -> None:
        self.pending.append(event)

    def on_complete(self, event: object) -> None:
        self.complete.append(event)


async def test_bridged_event_records_the_client_requested_model() -> None:
    """An alias-resolved request keeps the name the client actually asked for.

    Codex hardcodes the `codex-auto-review` guardian slug and the bridge maps it
    onto the eval model, so without this the reviewer's turns are recorded as
    the agent's own and reviewer-vs-agent attribution is impossible.
    """
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.types import AgentBridge
    from inspect_ai.agent._bridge.util import bridge_generate
    from inspect_ai.model._generate_config import GenerateConfig
    from inspect_ai.model._model import BRIDGE_REQUESTED_MODEL

    sink = _CollectingSink()
    bridge = AgentBridge(AgentState(messages=[]), model_event_sink=sink)
    model = get_model("mockllm/model")

    await bridge_generate(
        bridge,
        model,
        [],
        [],
        None,
        GenerateConfig(),
        requested_model="codex-auto-review",
    )

    assert sink.complete, "bridged generate emitted no ModelEvent"
    event = sink.complete[-1]
    metadata = getattr(event, "metadata", None) or {}
    assert metadata.get(BRIDGE_REQUESTED_MODEL) == "codex-auto-review"


async def test_filter_answered_request_still_records_an_event() -> None:
    """Output a filter produced without generating stays traceable.

    The scaffold consumes it, so a grader reading the transcript would otherwise
    see an assistant message that no ModelEvent accounts for.
    """
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.types import AgentBridge
    from inspect_ai.agent._bridge.util import bridge_generate
    from inspect_ai.model._generate_config import GenerateConfig
    from inspect_ai.model._model import (
        BRIDGE_FILTER_SYNTHETIC,
        BRIDGE_REQUESTED_MODEL,
    )
    from inspect_ai.model._model_output import ModelOutput

    filtered = ModelOutput.from_content("mockllm/model", "answered by the filter")

    async def _filter(
        model_name: str,
        input: object,
        tools: object,
        tool_choice: object,
        config: object,
    ) -> ModelOutput:
        return filtered

    sink = _CollectingSink()
    bridge = AgentBridge(AgentState(messages=[]), model_event_sink=sink, filter=_filter)
    model = get_model("mockllm/model")

    output, _ = await bridge_generate(
        bridge,
        model,
        [],
        [],
        None,
        GenerateConfig(),
        requested_model="codex-auto-review",
    )

    assert output.completion == "answered by the filter"
    assert sink.complete, "filter-answered request emitted no ModelEvent"
    # Exactly one pending/complete pair: _record_model_interaction self-completes
    # when given output, so an extra explicit complete() would double-forward the
    # event through the sink (a duplicated assistant message downstream).
    assert len(sink.pending) == 1
    assert len(sink.complete) == 1
    event = sink.complete[-1]
    metadata = getattr(event, "metadata", None) or {}
    assert metadata.get(BRIDGE_FILTER_SYNTHETIC) is True
    assert metadata.get(BRIDGE_REQUESTED_MODEL) == "codex-auto-review"
    # The event must carry the real output so a consumer can trace the
    # resulting assistant message back to it by content.
    assert getattr(event, "output").completion == "answered by the filter"
    assert getattr(event, "call") is None
