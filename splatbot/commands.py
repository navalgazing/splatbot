from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    def __init__(self, result: CommandResult) -> None:
        self.result = result
        detail = f"command failed ({result.returncode}): {' '.join(result.argv)}"
        if result.stdout:
            detail += f"\nstdout:\n{result.stdout[-4000:]}"
        if result.stderr:
            detail += f"\nstderr:\n{result.stderr[-4000:]}"
        super().__init__(detail)


class CommandRunner:
    def __init__(self, timeout_seconds: int | None = None, tail_bytes: int = 64 * 1024) -> None:
        self.timeout_seconds = timeout_seconds
        self.tail_bytes = tail_bytes

    async def run(
        self,
        argv: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        if _truthy_env("SPLATBOT_LOG_COMMAND_OUTPUT"):
            print(f"running command: {shlex.join(argv)}", flush=True)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr, returncode = await asyncio.wait_for(
                asyncio.gather(
                    _read_tail(proc.stdout, self.tail_bytes),
                    _read_tail(proc.stderr, self.tail_bytes),
                    proc.wait(),
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()
            result = CommandResult(
                argv=argv,
                returncode=-1,
                stdout="",
                stderr=f"command timed out after {self.timeout_seconds} seconds",
            )
            raise CommandError(result) from exc
        result = CommandResult(
            argv=argv,
            returncode=returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )
        if result.returncode != 0:
            raise CommandError(result)
        if _truthy_env("SPLATBOT_LOG_COMMAND_OUTPUT"):
            _print_command_output(result)
        return result


async def _read_tail(stream: asyncio.StreamReader | None, limit: int) -> bytes:
    if stream is None:
        return b""
    tail = bytearray()
    while chunk := await stream.read(8192):
        tail.extend(chunk)
        if len(tail) > limit:
            del tail[: len(tail) - limit]
    return bytes(tail)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def render_argv_template(command: str, values: Mapping[str, Any]) -> list[str]:
    text_values = {
        key: str(value)
        for key, value in values.items()
        if not isinstance(value, (list, tuple))
    }
    argv: list[str] = []
    for token in shlex.split(command):
        expanded_list = False
        for key, value in values.items():
            if token != f"{{{key}}}" or not isinstance(value, (list, tuple)):
                continue
            argv.extend(str(item) for item in value)
            expanded_list = True
            break
        if expanded_list:
            continue
        for key, value in values.items():
            if isinstance(value, (list, tuple)) and f"{{{key}}}" in token:
                raise ValueError(f"list placeholder {{{key}}} must be its own command token")
        try:
            argv.append(token.format(**text_values))
        except KeyError as exc:
            raise ValueError(f"unknown command placeholder: {exc.args[0]}") from exc
    return argv


def _print_command_output(result: CommandResult) -> None:
    print(f"command completed: {shlex.join(result.argv)}", flush=True)
    if result.stdout.strip():
        print("stdout tail:", flush=True)
        print(result.stdout[-4000:], flush=True)
    if result.stderr.strip():
        print("stderr tail:", flush=True)
        print(result.stderr[-4000:], flush=True)
