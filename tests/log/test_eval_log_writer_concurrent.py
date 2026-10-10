"""Concurrency shape for `EvalLogWriter` (dispatch://AGENTC-2235).

A long-running process (hawk's Harbor runner) holds many `EvalLogWriter`s at
once -- one per trial, several trials concurrent on one asyncio event loop --
and a different component than the one that opened the log appends its events
as they complete, closing each sample with a score or as errored.
"""

import asyncio
from datetime import datetime, timezone

from inspect_ai.event import InfoEvent
from inspect_ai.log import EvalLogWriter, read_eval_log
from inspect_ai.log._log import EvalConfig, EvalDataset, EvalPlan, EvalSample, EvalStats
from inspect_ai.log._log import EvalSpec as _EvalSpec


def _spec(task: str) -> _EvalSpec:
    return _EvalSpec(
        created=datetime.now(timezone.utc).isoformat(),
        task=task,
        model="mockllm/model",
        dataset=EvalDataset(name="test", samples=1),
        config=EvalConfig(),
    )


async def _event_appender(writer: EvalLogWriter, sample_id: int) -> None:
    """A different component than the opener: only appends events."""
    writer.log_event(sample_id, 1, InfoEvent(data=f"event for {sample_id}"))


async def _run_trial(tmp_path, trial: str, *, error: bool) -> None:
    """One trial's component.

    Opens the log, starts the sample, then hands the writer to a different
    component (`_event_appender`) before closing the sample with a score or
    an error.
    """
    location = str(tmp_path / f"{trial}.eval")
    writer = EvalLogWriter(location, _spec(trial))
    await writer.start(EvalPlan())

    summary = EvalSample(id=1, epoch=1, input="input", target="", messages=[]).summary()
    writer.start_sample(summary)

    # a different component appends the event
    await _event_appender(writer, 1)

    if error:
        from inspect_ai._util.error import EvalError

        sample = EvalSample(
            id=1,
            epoch=1,
            input="input",
            target="",
            messages=[],
            error=EvalError(message="boom", traceback="", traceback_ansi=""),
        )
    else:
        from inspect_ai.scorer._metric import Score

        sample = EvalSample(
            id=1,
            epoch=1,
            input="input",
            target="",
            messages=[],
            scores={"scorer": Score(value="C")},
        )
    await writer.complete_sample(sample)
    await writer.finish("success", EvalStats())

    read_back = read_eval_log(location)
    samples = read_back.samples
    assert samples is not None and len(samples) == 1
    if error:
        assert samples[0].error is not None
    else:
        assert samples[0].scores is not None


async def test_concurrent_trials_on_one_event_loop(tmp_path) -> None:
    # several trials concurrent on one event loop, one scored and one errored
    await asyncio.gather(
        _run_trial(tmp_path, "trial_a", error=False),
        _run_trial(tmp_path, "trial_b", error=True),
        _run_trial(tmp_path, "trial_c", error=False),
    )
