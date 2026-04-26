from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from dataclasses import dataclass
from pathlib import Path


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

    async def run(self, argv: list[str], cwd: Path | None = None) -> CommandResult:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.gather(
                    _read_tail(proc.stdout, self.tail_bytes),
                    _read_tail(proc.stderr, self.tail_bytes),
                ),
                timeout=self.timeout_seconds,
            )
            returncode = await proc.wait()
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
