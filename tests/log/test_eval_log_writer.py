"""Tests for `EvalLogWriter` (AGENTC-1744).

A caller running its own agent loop outside Inspect's eval loop -- for example
one trial of an external harness -- needs a public way to write a real `.eval`
log incrementally, with the same live sample-buffer progress a live Inspect
eval produces, without constructing a `Task`/`Dataset`/`Model` around it.
"""

from datetime import datetime, timezone

from inspect_ai.event import InfoEvent
from inspect_ai.log import EvalLogWriter, read_eval_log
from inspect_ai.log._log import (
    EvalConfig,
    EvalDataset,
    EvalPlan,
    EvalSample,
    EvalStats,
)
from inspect_ai.log._log import EvalSpec as _EvalSpec
from inspect_ai.log._recorders.buffer.buffer import sample_buffer


def _spec(task: str) -> _EvalSpec:
    return _EvalSpec(
        created=datetime.now(timezone.utc).isoformat(),
        task=task,
        model="mockllm/model",
        dataset=EvalDataset(name="test", samples=1),
        config=EvalConfig(),
    )


async def test_eval_log_writer_requires_start_before_use(tmp_path) -> None:
    # red: using the writer before start() must fail loudly, not silently no-op
    writer = EvalLogWriter(str(tmp_path / "trial.eval"), _spec("not_started"))
    try:
        writer.start_sample(
            EvalSample(id=1, epoch=1, input="input", target="", messages=[]).summary()
        )
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


async def test_eval_log_writer_streams_live_and_finishes(tmp_path) -> None:
    location = str(tmp_path / "trial.eval")
    spec = _spec("harbor_trial")
    writer = EvalLogWriter(location, spec)

    await writer.start(EvalPlan())

    sample_summary = EvalSample(
        id=1, epoch=1, input="input", target="target", messages=[]
    ).summary()
    writer.start_sample(sample_summary)
    writer.log_event(1, 1, InfoEvent(data="first"))
    writer.log_event(1, 1, InfoEvent(data="second"))

    # live: a second, independent read-only handle on the same location sees
    # the running sample and its events before the writer completes it --
    # exactly the progress a viewer polling `location` mid-eval would see.
    reader = sample_buffer(location)
    try:
        samples = reader.get_samples()
        assert samples != "NotModified" and samples is not None
        assert (sample_summary.id, sample_summary.epoch) in {
            (s.id, s.epoch) for s in samples.samples
        }
        assert reader.sample_event_count(1, 1) == 2
    finally:
        if hasattr(reader, "close"):
            reader.close()

    completed = EvalSample(
        id=1,
        epoch=1,
        input="input",
        target="target",
        messages=[],
        events=[InfoEvent(data="first"), InfoEvent(data="second")],
    )
    await writer.complete_sample(completed)

    # the completed sample is superseded in the live buffer -- the recorder's
    # flushed copy is now the record, not the (now-dropped) running entry
    reader = sample_buffer(location)
    try:
        samples = reader.get_samples()
        assert samples != "NotModified"
        if samples is not None:
            assert (1, 1) not in {(s.id, s.epoch) for s in samples.samples}
    finally:
        if hasattr(reader, "close"):
            reader.close()

    log = await writer.finish("success", EvalStats())
    assert log.status == "success"

    # the finished log is a real, independently readable .eval file
    read_back = read_eval_log(location)
    assert read_back.status == "success"
    read_back_samples = read_back.samples
    assert read_back_samples is not None and len(read_back_samples) == 1
    assert read_back_samples[0].id == 1


async def test_eval_log_writer_finish_twice_raises(tmp_path) -> None:
    location = str(tmp_path / "trial.eval")
    writer = EvalLogWriter(location, _spec("finish_twice"))
    await writer.start(EvalPlan())
    await writer.finish("success", EvalStats())
    try:
        await writer.finish("success", EvalStats())
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


async def test_eval_log_writer_discard() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        location = f"{tmp}/discarded.eval"
        writer = EvalLogWriter(location, _spec("discarded"))
        await writer.start(EvalPlan())
        writer.start_sample(
            EvalSample(id=1, epoch=1, input="input", target="", messages=[]).summary()
        )
        # a discarded log is never finished -- no exception, and a later
        # finish()/discard() is a no-op rather than a double-teardown error
        await writer.discard()
        await writer.discard()
