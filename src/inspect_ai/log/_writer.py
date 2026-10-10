"""A public API for writing one eval log incrementally, outside the eval loop.

See `EvalLogWriter`.
"""

from __future__ import annotations

from typing import Literal

from inspect_ai._util.error import EvalError
from inspect_ai._util.file import dirname
from inspect_ai.event import Event

from ._log import (
    EvalLog,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSampleReductions,
    EvalSampleSummary,
    EvalSpec,
    EvalStats,
    EvalStatus,
)
from ._recorders.buffer.database import SampleBufferDatabase
from ._recorders.create import create_recorder_for_format, create_recorder_for_location
from ._recorders.recorder import Recorder
from ._recorders.types import SampleEvent


class EvalLogWriter:
    """Writes one eval log incrementally, the way a live eval writes its own.

    Inspect's own eval loop (`inspect_ai._eval.task.log.TaskLogger`) opens a log,
    records its plan, then for each sample starts it, streams its events into a
    live sample buffer (synced to a remote log location when one is configured,
    so a log viewer or the control channel can show the sample's progress while it
    runs), records its finished result, and finally closes the log with the eval's
    overall status and stats. Everything that drives this is private
    (`inspect_ai.log._recorders.recorder.Recorder`,
    `inspect_ai.log._recorders.buffer.database.SampleBufferDatabase`), and reachable
    only from inside `TaskLogger`'s own eval-loop-specific construction.

    `EvalLogWriter` exposes the same sequence (`start`, `start_sample`/`log_event`
    per sample, `complete_sample`, `finish`) as a public API for a caller running
    its own agent loop outside Inspect's eval loop entirely -- for example, one
    trial of an external harness that wants to produce a real `.eval` (or `.json`)
    log with live progress, without running an Inspect `Task`/`Dataset`/`Model`
    around it. The caller supplies its own `EvalSpec`, `EvalPlan`, `EvalSample`,
    `EvalStatus` and `EvalStats` -- already-public types -- instead of this writer
    deriving them the way `TaskLogger` does from a live eval's `Task`.

    Open one writer per eval: construct it, call `start()` once, drive one or more
    samples through `start_sample()`/`log_event()`/`complete_sample()`, then
    `finish()` once. Call `discard()` instead of `finish()` if the eval ends
    before every result is known (an unrecoverable error partway through).
    """

    def __init__(
        self,
        location: str,
        eval: EvalSpec,
        *,
        format: Literal["eval", "json"] | None = None,
        log_images: bool = True,
        log_shared: int | None = None,
    ) -> None:
        """Create a writer for one eval log.

        Args:
          location: Where the log is written. Any location Inspect's own
            recorders support, including `s3://` and other `fsspec` schemes.
          eval: The eval's header. Caller-constructed; this writer does not
            derive it from a `Task`/`Dataset`/`Model`.
          format: The log format. Inferred from `location`'s extension
            (`.eval` or `.json`) when not given, matching Inspect's own
            `--log-format` resolution.
          log_images: Whether to include image attachments in the live sample
            buffer, as `TaskLogger`'s own `log_images` config does.
          log_shared: Sync interval (seconds) for a shared/remote log
            location's live segments, as `EvalConfig.log_shared` does. `None`
            (the default) keeps the sample buffer local only.
        """
        self.location = location
        self.eval = eval
        log_dir = dirname(location)
        self._recorder: Recorder = (
            create_recorder_for_format(format, log_dir)
            if format is not None
            else create_recorder_for_location(location, log_dir)
        )
        self._log_images = log_images
        self._log_shared = log_shared
        self._buffer_db: SampleBufferDatabase | None = None
        self._started = False
        self._finished = False

    async def start(self, plan: EvalPlan) -> None:
        """Open the log and record its plan.

        Call once, before any sample. Matches `TaskLogger.log_start`: writes
        the eval header and plan, then flushes immediately so the destination
        carries them as soon as the log starts -- the first progress a reader
        polling `location` would see.

        Args:
          plan: The eval's plan (solver/scorer steps), as recorded in the log.
        """
        if self._started:
            raise RuntimeError("EvalLogWriter.start() called more than once")
        await self._recorder.log_init(self.eval, self.location)
        await self._recorder.log_start(self.eval, plan)
        await self._recorder.flush(self.eval)
        self._buffer_db = SampleBufferDatabase(
            location=self.location,
            log_images=self._log_images,
            log_shared=self._log_shared,
        )
        self._started = True

    def start_sample(self, sample: EvalSampleSummary) -> None:
        """Record that a sample has begun, for live progress.

        Args:
          sample: The sample's summary (id, epoch, input) at the start of
            its run; its full result is not yet known.
        """
        buffer_db = self._require_started()
        buffer_db.start_sample(sample)

    def log_event(self, id: str | int, epoch: int, event: Event) -> None:
        """Append one event to a started sample's live transcript.

        Call this as the sample's agent loop produces events (model calls,
        tool calls, ...), the same way Inspect's own eval loop streams each
        event into the sample buffer as it happens.

        Args:
          id: The sample's id (matching an earlier `start_sample()` call).
          epoch: The sample's epoch.
          event: The event to append.
        """
        buffer_db = self._require_started()
        buffer_db.log_events([SampleEvent(id=id, epoch=epoch, event=event)])

    async def complete_sample(self, sample: EvalSample) -> None:
        """Record a sample's finished result, and remove it from the live buffer.

        Matches `TaskLogger.complete_sample`: the sample's full result is
        written through the recorder (its durable copy), flushed immediately
        (one writer serves one trial, so there is no batching threshold worth
        waiting on), and only then dropped from the live buffer -- the
        now-flushed recorder copy supersedes it.

        Args:
          sample: The sample's complete result (messages, output, score,
            events -- everything `start_sample()`/`log_event()` do not
            already carry, which this call supersedes).
        """
        buffer_db = self._require_started()
        await self._recorder.log_sample(self.eval, sample)
        buffer_db.complete_sample(sample.summary(), sample_metadata=sample.metadata)
        await self._recorder.flush(self.eval)
        buffer_db.remove_samples([(sample.id, sample.epoch)])

    async def finish(
        self,
        status: EvalStatus,
        stats: EvalStats,
        *,
        results: EvalResults | None = None,
        reductions: list[EvalSampleReductions] | None = None,
        error: EvalError | None = None,
    ) -> EvalLog:
        """Finalize and close the log. Matches `TaskLogger.log_finish`.

        Call once, after every sample has been completed. Tears down the live
        sample buffer; the finished log on disk/remote is the durable record
        from here on.

        Args:
          status: The eval's overall status.
          stats: The eval's overall stats.
          results: The eval's aggregate results (metrics), when scored.
          reductions: Per-sample score reductions, when the eval used
            multiple epochs with a reducer.
          error: The eval's fatal error, for a `status="error"` log.

        Returns:
          The finished `EvalLog`.
        """
        self._require_started()
        if self._finished:
            raise RuntimeError("EvalLogWriter.finish() called more than once")
        log = await self._recorder.log_finish(
            self.eval, status, stats, results, reductions, error=error
        )
        await self._teardown_buffer()
        self._finished = True
        return log

    async def discard(self) -> None:
        """Discard a log that will never be finished.

        Call instead of `finish()` when the eval ends before every result is
        known (an unrecoverable error partway through a trial). Matches
        `Recorder.log_discard`. A no-op if the writer was never started, or
        has already been finished or discarded.
        """
        if not self._started or self._finished:
            return
        await self._recorder.log_discard(self.eval)
        await self._teardown_buffer()
        self._finished = True

    async def _teardown_buffer(self) -> None:
        if self._buffer_db is not None:
            await self._buffer_db.aclose()
            self._buffer_db = None

    def _require_started(self) -> SampleBufferDatabase:
        if not self._started:
            raise RuntimeError("EvalLogWriter.start() must be called first")
        if self._finished:
            raise RuntimeError("EvalLogWriter already finished/discarded")
        assert self._buffer_db is not None
        return self._buffer_db
