import pytest

from splatbot.commands import CommandError, CommandRunner, render_argv_template


def test_render_argv_template_preserves_paths_with_spaces() -> None:
    argv = render_argv_template(
        "tool --input {input_dir} --flag={value}",
        {"input_dir": "/tmp/path with spaces", "value": "abc def"},
    )

    assert argv == ["tool", "--input", "/tmp/path with spaces", "--flag=abc def"]


def test_render_argv_template_expands_list_placeholder_as_args() -> None:
    argv = render_argv_template("tool {extra_args}", {"extra_args": ["--one", "two words"]})

    assert argv == ["tool", "--one", "two words"]


def test_render_argv_template_rejects_embedded_list_placeholder() -> None:
    with pytest.raises(ValueError, match="must be its own command token"):
        render_argv_template("tool --args={extra_args}", {"extra_args": ["--one"]})


async def test_command_runner_times_out_hung_process() -> None:
    runner = CommandRunner(timeout_seconds=0.1)

    with pytest.raises(CommandError, match="timed out"):
        await runner.run(["python3", "-c", "import time; time.sleep(5)"])
