from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import anyio

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import Generate, Solver, TaskState, solver


@solver
def _observe_resource(events: list[str]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        assert events == ["entered"]
        events.append("solver")
        return state

    return solve


@asynccontextmanager
async def _record_resource(events: list[str], _state: TaskState) -> AsyncIterator[None]:
    events.append("entered")
    try:
        yield
    finally:
        events.append("exited")


def test_sample_resource_wraps_the_sample_lifecycle() -> None:
    events: list[str] = []
    task = Task(
        dataset=[Sample(input="test")],
        solver=_observe_resource(events),
        sample_resources=[lambda state: _record_resource(events, state)],
    )

    eval(task, model="mockllm/model")

    assert events == ["entered", "solver", "exited"]


@asynccontextmanager
async def _task_group_resource(
    events: list[str], _state: TaskState
) -> AsyncIterator[None]:
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(anyio.sleep_forever)
        try:
            yield
        finally:
            with anyio.move_on_after(2, shield=True):
                await anyio.lowlevel.checkpoint()
                events.append("teardown")
            task_group.cancel_scope.cancel()


async def test_sample_resource_cleanup_preserves_cancel_scope_order() -> None:
    from inspect_ai._eval.task.run import _sample_resources_cm

    events: list[str] = []
    entered = anyio.Event()

    async def sample_body() -> None:
        async with _sample_resources_cm(
            [lambda state: _task_group_resource(events, state)],
            cast(TaskState, None),
        ):
            entered.set()
            await anyio.sleep_forever()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(sample_body)
        await entered.wait()
        task_group.cancel_scope.cancel()

    assert events == ["teardown"]
