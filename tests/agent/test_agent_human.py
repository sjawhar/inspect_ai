import concurrent.futures
import re
import subprocess
import sys
import threading
import time
from argparse import Namespace
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from io import StringIO
from pathlib import Path
from typing import AsyncIterator, Callable, Iterator, override

import anyio
import pytest
from test_helpers.utils import skip_if_no_docker

from inspect_ai import Task, eval
from inspect_ai.agent import (
    AgentState,
    HumanAgentCommand,
    HumanAgentCommandsFilter,
    human_cli,
)
from inspect_ai.agent._human import agent as human_agent
from inspect_ai.agent._human import service as human_service
from inspect_ai.agent._human.commands import human_agent_commands, submit
from inspect_ai.agent._human.commands.instructions import InstructionsCommand
from inspect_ai.agent._human.commands.submit import QuitCommand, SubmitCommand
from inspect_ai.agent._human.install import (
    human_agent_commands as generated_human_agent_commands,
)
from inspect_ai.agent._human.state import HumanAgentState
from inspect_ai.util import sandbox

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


@pytest.mark.parametrize(
    ("command", "args", "expected_calls"),
    [
        (QuitCommand(False), Namespace(), []),
        (
            SubmitCommand(False),
            Namespace(answer=None),
            [("validate", {"answer": None})],
        ),
    ],
)
def test_session_end_commands_decline_on_eof(
    command: QuitCommand | SubmitCommand,
    args: Namespace,
    expected_calls: list[tuple[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def call_human_agent(method: str, **params: object) -> None:
        calls.append((method, params))

    monkeypatch.setattr(submit, "call_human_agent", call_human_agent)
    monkeypatch.setattr(sys, "stdin", StringIO())

    command.cli(args)

    assert calls == expected_calls


class _AdditionalCommand(HumanAgentCommand):
    @property
    def name(self) -> str:
        return "additional"

    @property
    def description(self) -> str:
        return "Additional test command."


class _OverrideCommand(HumanAgentCommand):
    @property
    def name(self) -> str:
        return "override"

    @property
    def description(self) -> str:
        return "A command whose handler uses a type-only decorator."

    @override
    def cli(self, args: Namespace) -> None:
        """The generated handler documentation contains @override

        and is emitted by this custom command.
        """
        # @override
        message = """handler literal @override
        is preserved."""
        del args
        override_doc = override.__doc__
        assert override_doc is not None
        print(override_doc.splitlines()[0])
        print(message.splitlines()[0])


def test_generated_human_agent_commands_execute_override_handler(
    tmp_path: Path,
) -> None:
    task_py = generated_human_agent_commands([_OverrideCommand()])
    (tmp_path / "human_agent.py").write_text(
        "def call_human_agent(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    task_py_path = tmp_path / "task.py"
    task_py_path.write_text(task_py, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(task_py_path), "override"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "The generated handler documentation contains @override\n"
        "handler literal @override\n"
    )
    assert "# @override\n" in task_py


def test_human_cli_accepts_public_commands_filter():
    def commands_filter(
        commands: list[HumanAgentCommand],
    ) -> list[HumanAgentCommand]:
        return [*commands, _AdditionalCommand()]

    filter_: HumanAgentCommandsFilter = commands_filter

    assert callable(human_cli(commands_filter=filter_))


async def test_human_cli_commands_filter_seen_by_instructions() -> None:
    def commands_filter(
        commands: list[HumanAgentCommand],
    ) -> list[HumanAgentCommand]:
        return [*commands, _AdditionalCommand()]

    commands = human_agent_commands(
        AgentState(messages=[]),
        answer=True,
        intermediate_scoring=False,
        record_session=False,
        instructions=None,
        commands_filter=commands_filter,
    )

    # the filter's appended command is in the built list, ahead of the
    # instructions command that the filter must run before
    names = [command.name for command in commands]
    assert names.index("additional") < names.index("instructions")

    # and the instructions command itself was built from the filtered list,
    # so `task instructions` renders the added command
    instructions_command = commands[-1]
    assert isinstance(instructions_command, InstructionsCommand)
    rendered = await instructions_command.service(
        HumanAgentState(instructions="do the task")
    )()
    assert isinstance(rendered, str)
    assert "additional" in rendered
    assert "Additional test command." in rendered


@pytest.mark.slow
@skip_if_no_docker
@pytest.mark.parametrize("user", ["root", "nonroot", None])
def test_human_cli(user: str | None) -> None:
    ready = threading.Event()
    closed = threading.Event()
    container_name: str | None = None

    @asynccontextmanager
    async def on_ready() -> AsyncIterator[None]:
        nonlocal container_name
        connection = await sandbox().connection(user=user)
        assert connection.container is not None
        container_name = connection.container
        ready.set()
        try:
            yield
        finally:
            closed.set()

    def run_eval():
        task = Task(
            solver=human_cli(user=user, on_ready=on_ready),
            sandbox=(
                "docker",
                (Path(__file__).parent / "compose.human.yaml").as_posix(),
            ),
        )
        return eval(task, display="plain")[0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_eval)

        assert ready.wait(timeout=60)
        assert container_name is not None
        docker_exec = [
            "docker",
            "exec",
            *(["-u", user] if user else []),
            container_name,
            "bash",
            "-l",
            "-c",
        ]

        subprocess.check_call(docker_exec + ["python3 /opt/human_agent/task.py start"])
        assert not closed.is_set()
        subprocess.check_call(
            docker_exec
            + [
                'echo -e "y\\n" | python3 /opt/human_agent/task.py submit "test"',
            ],
        )

        done, _ = concurrent.futures.wait([future], timeout=20)
        if future in done:
            log = future.result()
            assert log.status == "success"
            assert log.samples[0].output.choices[0].message.content == "test"
            assert closed.is_set()
        else:
            raise Exception("eval() did not complete within timeout")


async def test_human_cli_connects_view_after_ready_context_enters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeSandbox:
        async def connection(self, *, user: str | None) -> object:
            return object()

        @contextmanager
        def no_events(self) -> Iterator[None]:
            yield

    class FakeConsoleView:
        def connect(self, connection: object) -> None:
            events.append("view-connected")

        def update_state(self, state: object) -> None:
            pass

    @asynccontextmanager
    async def on_ready() -> AsyncIterator[None]:
        events.append("ready-entered")
        try:
            yield
        finally:
            events.append("ready-exited")

    async def fake_install_human_agent(*_: object) -> None:
        events.append("installed")

    async def fake_run_human_agent_service(
        user: str | None,
        state: AgentState,
        commands: list[object],
        view: object,
        ready: Callable[[], AbstractAsyncContextManager[None]] | None,
    ) -> AgentState:
        assert ready is not None
        events.append("service-started")
        async with ready():
            events.append("service-running")
        events.append("service-finished")
        return state

    monkeypatch.setattr(human_agent, "sandbox", lambda: FakeSandbox())
    monkeypatch.setattr(human_agent, "display_type", lambda: "plain")
    monkeypatch.setattr(human_agent, "ConsoleView", FakeConsoleView)
    monkeypatch.setattr(human_agent, "install_human_agent", fake_install_human_agent)
    monkeypatch.setattr(
        human_agent, "run_human_agent_service", fake_run_human_agent_service
    )

    await human_cli(on_ready=on_ready)(AgentState(messages=[]))

    assert events == [
        "installed",
        "service-started",
        "ready-entered",
        "view-connected",
        "service-running",
        "ready-exited",
        "service-finished",
    ]


async def test_human_cli_ready_context_waits_for_service_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_entered = anyio.Event()

    class ServiceFailure(Exception):
        pass

    async def fake_sandbox_service(**_: object) -> None:
        raise ServiceFailure("human service failed before startup")

    @asynccontextmanager
    async def on_ready() -> AsyncIterator[None]:
        context_entered.set()
        yield

    def no_clock_action_event(*_: object) -> None:
        return None

    monkeypatch.setattr(human_service, "clock_action_event", no_clock_action_event)
    monkeypatch.setattr(human_service, "sandbox", object)
    monkeypatch.setattr(human_service, "sandbox_service", fake_sandbox_service)

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await human_service.run_human_agent_service(
            None, AgentState(messages=[]), [], None, on_ready=on_ready
        )

    assert any(isinstance(error, ServiceFailure) for error in exc_info.value.exceptions)
    assert not context_entered.is_set()


async def test_human_cli_without_ready_preserves_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ServiceFailure(Exception):
        pass

    async def fake_sandbox_service(**_: object) -> None:
        raise ServiceFailure("human service failed before startup")

    def no_clock_action_event(*_: object) -> None:
        return None

    monkeypatch.setattr(human_service, "clock_action_event", no_clock_action_event)
    monkeypatch.setattr(human_service, "sandbox", object)
    monkeypatch.setattr(human_service, "sandbox_service", fake_sandbox_service)

    with pytest.raises(ServiceFailure, match="before startup"):
        await human_service.run_human_agent_service(
            None, AgentState(messages=[]), [], None
        )


async def test_human_cli_ready_context_closes_after_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_entered = anyio.Event()
    context_exited = anyio.Event()

    class ServiceFailure(Exception):
        pass

    async def fake_sandbox_service(
        *, started: anyio.Event | None = None, **_: object
    ) -> None:
        assert started is not None
        started.set()
        await context_entered.wait()
        raise ServiceFailure("human service failed")

    @asynccontextmanager
    async def on_ready() -> AsyncIterator[None]:
        context_entered.set()
        try:
            yield
        finally:
            context_exited.set()

    def no_clock_action_event(*_: object) -> None:
        return None

    monkeypatch.setattr(human_service, "clock_action_event", no_clock_action_event)
    monkeypatch.setattr(human_service, "sandbox", object)
    monkeypatch.setattr(human_service, "sandbox_service", fake_sandbox_service)

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await human_service.run_human_agent_service(
            None, AgentState(messages=[]), [], None, on_ready=on_ready
        )

    assert any(isinstance(error, ServiceFailure) for error in exc_info.value.exceptions)
    assert context_exited.is_set()


async def test_human_cli_ready_context_closes_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_entered = anyio.Event()
    context_exited = anyio.Event()

    async def fake_sandbox_service(
        *, started: anyio.Event | None = None, **_: object
    ) -> None:
        assert started is not None
        started.set()
        await context_entered.wait()
        await anyio.sleep_forever()

    @asynccontextmanager
    async def on_ready() -> AsyncIterator[None]:
        context_entered.set()
        try:
            yield
        finally:
            context_exited.set()

    def no_clock_action_event(*_: object) -> None:
        return None

    monkeypatch.setattr(human_service, "clock_action_event", no_clock_action_event)
    monkeypatch.setattr(human_service, "sandbox", object)
    monkeypatch.setattr(human_service, "sandbox_service", fake_sandbox_service)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            human_service.run_human_agent_service,
            None,
            AgentState(messages=[]),
            [],
            None,
            on_ready,
        )
        await context_entered.wait()
        task_group.cancel_scope.cancel()

    assert context_exited.is_set()


@pytest.mark.slow
@skip_if_no_docker
def test_human_cli_submit_no_answer(capsys: pytest.CaptureFixture[str]):
    """Test that submitting without an answer completes the task when answer=False."""

    def run_eval():
        task = Task(
            solver=human_cli(answer=False),
            sandbox=(
                "docker",
                (Path(__file__).parent / "compose.human.yaml").as_posix(),
            ),
        )
        return eval(task, display="plain")[0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_eval)

        out = ""
        container_name = None
        for _ in range(10):
            out += capsys.readouterr().out
            if match := re.search(r"inspect-task-\S+-default-1", out):
                container_name = match.group(0)
                break
            time.sleep(1)

        if not container_name:
            raise Exception("Failed to find container name")

        docker_exec = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-l",
            "-c",
        ]

        human_agent_found = False
        for _ in range(10):
            result = subprocess.run(
                docker_exec
                + ["ls /var/tmp/sandbox-services/human_agent/human_agent.py"]
            )
            if result.returncode == 0:
                human_agent_found = True
                break
            time.sleep(1)

        if not human_agent_found:
            raise Exception("Human agent sandbox service not found")

        subprocess.check_call(docker_exec + ["python3 /opt/human_agent/task.py start"])
        # Submit without an answer - this should complete the task when answer=False
        subprocess.check_call(
            docker_exec
            + [
                'echo -e "y\\n" | python3 /opt/human_agent/task.py submit',
            ],
        )

        done, _ = concurrent.futures.wait([future], timeout=5)
        if future in done:
            log = future.result()
            assert log.status == "success"
            assert log.samples[0].output.choices[0].message.content == ""
        else:
            raise Exception("eval() did not complete within timeout")
