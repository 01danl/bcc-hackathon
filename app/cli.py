"""Command-line interface of the requirements reviewer (Typer)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from app.config import Settings, SettingsError, load_settings
from app.ingestion.document_loader import DocumentLoader, DocumentLoaderError
from app.llm.client import (
    LLMConfigurationError,
    LLMError,
    build_llm_client,
)
from app.models.analysis import (
    RequirementAnalysisCollection,
    RequirementStatus,
    Severity,
)
from app.models.requirement import (
    DocumentFormat,
    Requirement,
    RequirementCollection,
    SourceDocument,
)
from app.services.chunking import DocumentChunker
from app.services.requirement_analyzer import RequirementAnalyzer
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
    save_json: Annotated[
        bool,
        typer.Option(
            "--save-json/--no-save-json",
            help="Сохранять промежуточные результаты извлечения и анализа в JSON.",
        ),
    ] = True,
) -> None:
    """Извлечь требования и проверить их с помощью LLM."""
    try:
        settings = load_settings()
    except SettingsError as exc:
        _fail(f"Ошибка конфигурации: {exc}")

    try:
        llm_client = build_llm_client(settings)
    except LLMConfigurationError as exc:
        _fail(f"Ошибка конфигурации LLM: {exc}")

    loader = DocumentLoader()
    try:
        document = loader.load(input_path)
    except DocumentLoaderError as exc:
        _fail(f"Ошибка загрузки документа: {exc}")

    chunker = DocumentChunker(
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    chunks = chunker.split(document)
    extractor = RequirementExtractor(llm_client, chunker)
    analyzer = RequirementAnalyzer(llm_client)

    typer.secho(
        f"Извлечение требований с помощью LLM ({settings.llm_model})...",
        bold=True,
    )
    try:
        collection = extractor.extract(document)
    except LLMError as exc:
        _fail(f"Ошибка LLM при извлечении требований: {exc}")

    for warning in extractor.validate(collection):
        typer.secho(f"Предупреждение: {warning}", fg=typer.colors.YELLOW, err=True)

    extraction_path = _json_path(output_path, "requirements")
    if save_json:
        _write_json(
            extraction_path,
            {
                "passport": _passport(settings, document, len(chunks)),
                "requirements": [
                    requirement.model_dump(mode="json")
                    for requirement in collection.requirements
                ],
            },
        )
        typer.echo(f"Результат извлечения сохранён: {extraction_path}")

    typer.secho(
        f"Извлечено требований: {len(collection)}. Запуск анализа...",
        bold=True,
    )
    try:
        analyses = analyzer.analyze_all(
            collection,
            document_text=document.text,
            progress=_print_progress,
        )
    except LLMError as exc:
        if save_json:
            typer.secho(
                f"Промежуточный результат извлечения сохранён: {extraction_path}",
                fg=typer.colors.YELLOW,
                err=True,
            )
        _fail(f"Ошибка LLM при анализе требований: {exc}")

    analysis_path = _json_path(output_path, "analysis")
    if save_json:
        _write_json(
            analysis_path,
            {
                "passport": _passport(settings, document, len(chunks)),
                "summary": _summary(analyses),
                "analyses": [
                    analysis.model_dump(mode="json")
                    for analysis in analyses.analyses
                ],
            },
        )

    _write_report(
        output_path,
        document=document,
        collection=collection,
        analyses=analyses,
        settings=settings,
        chunk_count=len(chunks),
    )

    _print_summary(
        output_path=output_path,
        extraction_path=extraction_path if save_json else None,
        analysis_path=analysis_path if save_json else None,
        collection=collection,
        analyses=analyses,
    )


def _print_progress(number: int, total: int, requirement: Requirement) -> None:
    """Print one line per analyzed requirement."""
    typer.echo(f"  [{number}/{total}] {requirement.id}")


def _print_summary(
    *,
    output_path: Path,
    extraction_path: Path | None,
    analysis_path: Path | None,
    collection: RequirementCollection,
    analyses: RequirementAnalysisCollection,
) -> None:
    """Print a short execution summary."""
    status_counts = analyses.by_status()
    typer.secho("Готово.", bold=True)
    typer.echo(f"  Требований: {len(collection)}")
    for status in RequirementStatus:
        typer.echo(f"  {status.value}: {status_counts.get(status, 0)}")
    typer.echo(f"  Находок: {analyses.finding_count}")
    typer.echo(f"  Отчёт: {output_path}")
    if extraction_path is not None:
        typer.echo(f"  JSON извлечения: {extraction_path}")
    if analysis_path is not None:
        typer.echo(f"  JSON анализа: {analysis_path}")


def _passport(
    settings: Settings, document: SourceDocument, chunk_count: int
) -> dict[str, object]:
    """Build the run passport shared by JSON files and the report."""
    return {
        "document": str(document.path),
        "format": document.format.value,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "agent": settings.agent_name,
        "agent_version": settings.agent_version,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "chunks": chunk_count,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write UTF-8 JSON with a stable, human-readable layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_path(output_path: Path, suffix: str) -> Path:
    """Derive a sidecar JSON path from the report path."""
    return output_path.parent / f"{output_path.stem}.{suffix}.json"


def _summary(analyses: RequirementAnalysisCollection) -> dict[str, object]:
    """Build machine-readable status and severity counters."""
    status_counter = Counter(analysis.status for analysis in analyses.analyses)
    severity_counter: Counter[Severity] = Counter()
    for analysis in analyses.analyses:
        severity_counter.update(finding.severity for finding in analysis.findings)
    return {
        "requirements": len(analyses.analyses),
        "findings": analyses.finding_count,
        "status": {
            status.value: status_counter.get(status, 0)
            for status in RequirementStatus
        },
        "severity": {
            severity.value: severity_counter.get(severity, 0)
            for severity in Severity
        },
    }


def _write_report(
    output: Path,
    *,
    document: SourceDocument,
    collection: RequirementCollection,
    analyses: RequirementAnalysisCollection,
    settings: Settings,
    chunk_count: int,
) -> None:
    """Write a plain Markdown report with structured LLM results.

    This is deliberately not the final polished report: it exposes the
    extraction and per-requirement findings so the run can be verified.
    """
    passport = _passport(settings, document, chunk_count)
    summary = _summary(analyses)
    lines: list[str] = [
        "# Отчёт проверки требований (LLM-анализ)",
        "",
        "> Отчёт сформирован автоматически по результатам реального вызова LLM.",
        "> Финальная редакторская версия отчёта и проверка набора требований",
        "> целиком — следующий этап проекта.",
        "",
        "## Паспорт прогона",
        "",
        "| Параметр | Значение |",
        "| :--- | :--- |",
        f"| Документ | {_cell(Path(str(passport['document'])).name)} |",
        f"| Формат | {_cell(str(passport['format']))} |",
        f"| Дата прогона | {_cell(str(passport['generated_at']))} |",
        f"| Агент | {_cell(str(passport['agent']))}, "
        f"версия {_cell(str(passport['agent_version']))} |",
        f"| LLM-модель | {_cell(str(passport['llm_model'] or '—'))} |",
        f"| LLM base URL | {_cell(str(passport['llm_base_url']))} |",
        f"| Чанков отправлено | {chunk_count} |",
        f"| Требований извлечено | {len(collection)} |",
        "",
        "## Нарезка требований",
        "",
        "| Идентификатор | Идентификатор в документе | Категория | Раздел / страница | Формулировка |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for requirement in collection.requirements:
        lines.append(
            f"| {_cell(requirement.id)} | {_cell(requirement.original_id or '—')} | "
            f"{_cell(requirement.category.value)} | "
            f"{_cell(_location_label(requirement))} | "
            f"{_cell(requirement.text)} |"
        )
    if not collection.requirements:
        lines.append("| — | — | — | — | Требования не извлечены |")

    lines.extend(["", "## Проверка каждого требования", ""])
    for analysis in analyses.analyses:
        lines.append(f"### {analysis.requirement_id} — {analysis.status.value}")
        lines.append("")
        if not analysis.findings:
            lines.append("Нарушений не найдено.")
            lines.append("")
            continue
        lines.extend(
            [
                "| Статус | Критерий | Severity | Проблема | Доказательство | Рекомендация |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for finding in analysis.findings:
            lines.append(
                f"| {finding.status.value} | {finding.criterion.value} | "
                f"{finding.severity.value} | {_cell(finding.explanation)} | "
                f"{_cell(finding.evidence)} | {_cell(finding.recommendation)} |"
            )
        lines.append("")

    if not analyses.analyses:
        lines.extend(["Требования для анализа отсутствуют.", ""])

    lines.extend(
        [
            "## Сводка",
            "",
            f"- Требований: {summary['requirements']}.",
            f"- Находок: {summary['findings']}.",
        ]
    )
    status_summary = summary["status"]
    severity_summary = summary["severity"]
    assert isinstance(status_summary, dict)
    assert isinstance(severity_summary, dict)
    for status in RequirementStatus:
        lines.append(f"- {status.value}: {status_summary[status.value]}.")
    for severity in Severity:
        lines.append(f"- Severity {severity.value}: {severity_summary[severity.value]}.")
    lines.append("")

    critical = [
        (analysis.requirement_id, finding)
        for analysis in analyses.analyses
        for finding in analysis.findings
        if finding.severity in {Severity.HIGH, Severity.CRITICAL}
    ]
    if critical:
        lines.extend(["## Критичные и высокие дефекты", ""])
        for requirement_id, finding in critical:
            lines.append(
                f"- **{requirement_id}** ({finding.severity.value}, "
                f"{finding.criterion.value}): {finding.explanation}"
            )
        lines.append("")

    lines.extend(
        [
            "## Ограничения текущего этапа",
            "",
            "- Проверяется каждое требование отдельно; consistency сверяется с",
            "  полным набором, но отдельный агрегированный анализ всего набора",
            "  (покрытие, дубликаты, полнота) ещё не реализован.",
            "- Итоговый отчёт не проходит редакторскую полировку и не содержит",
            "  финального вердикта о готовности к разработке.",
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _location_label(requirement: Requirement) -> str:
    """Human-readable source location for the report table."""
    location = requirement.location
    if location.section:
        return location.section
    if location.page is not None:
        return f"стр. {location.page}"
    if location.heading:
        return location.heading
    return "—"


def _cell(text: str) -> str:
    """Escape markdown table syntax in a single table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _fail(message: str) -> NoReturn:
    """Print an error and terminate with a non-zero exit code."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)
