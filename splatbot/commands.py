from __future__ import annotations

import asyncio
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
    async def run(self, argv: list[str], cwd: Path | None = None) -> CommandResult:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        result = CommandResult(
            argv=argv,
            returncode=proc.returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )
        if result.returncode != 0:
            raise CommandError(result)
        return result
