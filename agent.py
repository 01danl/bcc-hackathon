"""Root entry point for a one-command audit.

Usage::

    python agent.py --input docs/requirements.md --output report.md

The command intentionally has no subcommand: it delegates to the same
``app.cli.analyze`` pipeline that ``python -m app.main analyze`` uses, so the
full run (dotenv loading, extraction, static linting, triage, batch audit,
set audit and report writing) is identical. The legacy module entry point
keeps working unchanged.
"""

from __future__ import annotations

from pathlib import Path

import typer

from app.cli import analyze

agent_app = typer.Typer(add_completion=False, no_args_is_help=True)


@agent_app.command()
def main(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Путь к документу требований (.md или .pdf).",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        dir_okay=False,
        help="Путь к итоговому Markdown-отчёту.",
    ),
) -> None:
    """Провести полный аудит документа и записать отчёт."""
    # ``--output report.md`` has '.' as its parent; mkdir is harmless there.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyze(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    agent_app()
