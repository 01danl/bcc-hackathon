"""Command-line interface of the requirements reviewer (Typer)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from app.config import Settings, load_settings
from app.ingestion.document_loader import DocumentLoader, DocumentLoaderError
from app.models.requirement import (
    DocumentFormat,
    Requirement,
    RequirementCollection,
    SourceDocument,
)
from app.services.requirement_extractor import RequirementExtractor

app = typer.Typer(
    name="requirements-reviewer",
    help="Ревью документов с требованиями.",
    no_args_is_help=True,
)


@app.callback()
def _callback() -> None:
    """AI Requirements Reviewer — проверка качества требований."""


@app.command()
def analyze(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Путь к документу требований (.md, .markdown или .pdf).",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Путь к файлу отчёта (каталог создаётся автоматически).",
        ),
    ],
) -> None:
    """Загрузить документ, извлечь требования и записать предварительный отчёт."""
    settings = load_settings()

    loader = DocumentLoader()
    try:
        document = loader.load(input_path)
    except DocumentLoaderError as exc:
        typer.secho(f"Ошибка загрузки документа: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    collection = RequirementExtractor().extract(document)

    _print_stats(document, collection)
    _write_preliminary_report(output_path, document, collection, settings)
    typer.echo(f"Предварительный отчёт записан: {output_path}")


def _print_stats(document: SourceDocument, collection: RequirementCollection) -> None:
    """Print basic processing statistics to the console."""
    typer.secho("Статистика обработки (этап: фундамент, ИИ-анализ ещё не подключён)", bold=True)
    typer.echo(f"  Документ: {document.path} ({document.format.value})")
    if document.format is DocumentFormat.PDF:
        typer.echo(f"  Страниц: {document.page_count}")
    typer.echo(f"  Символов: {len(document.text)}")
    typer.echo(f"  Строк: {len(document.text.splitlines())}")
    typer.echo(f"  Кандидатов в требования: {len(collection)} (эвристика-заполнитель)")
    for category, items in sorted(collection.by_category().items()):
        typer.echo(f"    {category.value}: {len(items)}")


def _write_preliminary_report(
    output: Path,
    document: SourceDocument,
    collection: RequirementCollection,
    settings: Settings,
) -> None:
    """Write the preliminary placeholder report.

    This is not the final report: no requirement quality analysis has
    been performed yet. The real report generator arrives in a later
    milestone and will replace this function.
    """
    lines: list[str] = [
        "# Предварительный отчёт проверки требований",
        "",
        "> Статус: этап «фундамент» проекта. Выполнены загрузка документа и черновая",
        "> нарезка требований. ИИ-анализ формулировок (двусмысленность, неполнота,",
        "> противоречия, непроверяемость) на этом этапе **не выполнялся**.",
        "",
        "## Паспорт прогона",
        "",
        "| Параметр | Значение |",
        "| :--- | :--- |",
        f"| Документ | {_cell(document.path.name)} |",
        f"| Формат | {document.format.value} |",
        f"| Дата прогона | {datetime.now().strftime('%Y-%m-%d %H:%M')} |",
        f"| Агент | {settings.agent_name}, версия {settings.agent_version} |",
        f"| LLM-модель | {settings.llm_model or 'не подключена'} |",
        "",
        "## Статистика обработки",
        "",
        f"- Символов: {len(document.text)}.",
        f"- Строк: {len(document.text.splitlines())}.",
    ]
    if document.format is DocumentFormat.PDF:
        lines.append(f"- Страниц: {document.page_count}.")
    lines.extend(
        [
            f"- Выделено кандидатов в требования: {len(collection)}.",
            "",
            "## Нарезка требований (черновая эвристика)",
            "",
            "| Идентификатор | Идентификатор в документе | Раздел / страница | Формулировка |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )
    for requirement in collection.requirements:
        lines.append(
            f"| {requirement.id} | {requirement.original_id or '—'} | "
            f"{_cell(_location_label(requirement))} | {_cell(_shorten(requirement.text))} |"
        )
    lines.extend(
        [
            "",
            "## Что дальше",
            "",
            "На следующих этапах: извлечение требований с помощью LLM, проверка каждого",
            "требования и набора целиком, итоговый отчёт на русском языке.",
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _location_label(requirement: Requirement) -> str:
    """Human-readable source location for the report table."""
    if requirement.location.section:
        return requirement.location.section
    if requirement.location.page is not None:
        return f"стр. {requirement.location.page}"
    return "—"


def _shorten(text: str, limit: int = 100) -> str:
    """Truncate long requirement texts for the report table."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _cell(text: str) -> str:
    """Escape markdown table syntax in a single table cell."""
    return text.replace("|", "\\|").replace("\n", " ")
