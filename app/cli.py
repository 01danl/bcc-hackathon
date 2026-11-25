"""Command-line interface of the requirements reviewer (Typer)."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from app.config import Settings, SettingsError, load_settings, reasoning_extra_body
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
    Requirement,
    RequirementCollection,
    SourceDocument,
)
from app.models.set_audit import RequirementSetAudit, SetFinding, VerdictStatus
from app.services.chunking import DocumentChunker
from app.services.requirement_analyzer import RequirementAnalyzer
from app.services.requirement_extractor import RequirementExtractor
from app.services.set_analyzer import RequirementSetAnalyzer

# The agent's own code: git is queried here, and only changes under it
# mark the reported commit as dirty.
_AGENT_DIR = Path(__file__).resolve().parent

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
    """Извлечь требования, проверить каждое и весь набор с помощью LLM."""
    try:
        settings = load_settings()
    except SettingsError as exc:
        _fail(f"Ошибка конфигурации: {exc}")

    try:
        extractor_client = build_llm_client(
            settings.with_model(
                settings.extractor_model,
                extra_body=reasoning_extra_body(settings.extractor_effort),
                max_tokens=settings.extractor_max_tokens,
            )
        )
        auditor_client = build_llm_client(
            settings.with_model(
                settings.auditor_model,
                extra_body=reasoning_extra_body(settings.auditor_effort),
                max_tokens=settings.auditor_max_tokens,
            )
        )
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
    passport = _passport(settings, document, len(chunks))

    extractor = RequirementExtractor(extractor_client, chunker)
    analyzer = RequirementAnalyzer(auditor_client)
    set_analyzer = RequirementSetAnalyzer(auditor_client)

    typer.secho(
        f"Извлечение требований с помощью LLM ({settings.extractor_model})...",
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
                "passport": passport,
                "requirements": [
                    requirement.model_dump(mode="json")
                    for requirement in collection.requirements
                ],
            },
        )
        typer.echo(f"Результат извлечения сохранён: {extraction_path}")

    typer.secho(
        f"Извлечено требований: {len(collection)}. "
        f"Анализ каждого требования ({settings.auditor_model})...",
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
    analysis_payload: dict[str, object] = {
        "passport": passport,
        "summary": _summary(analyses),
        "analyses": [
            analysis.model_dump(mode="json") for analysis in analyses.analyses
        ],
    }
    if save_json:
        _write_json(analysis_path, analysis_payload)

    typer.secho("Проверка набора требований целиком...", bold=True)
    try:
        set_audit = set_analyzer.analyze(document.text, collection, analyses.analyses)
    except LLMError as exc:
        if save_json:
            typer.secho(
                "Промежуточные результаты сохранены: "
                f"{extraction_path}, {analysis_path}",
                fg=typer.colors.YELLOW,
                err=True,
            )
        _fail(f"Ошибка LLM при проверке набора требований: {exc}")

    if save_json:
        _write_json(
            analysis_path,
            {**analysis_payload, "set_audit": set_audit.model_dump(mode="json")},
        )

    _write_report(
        output_path,
        passport=passport,
        collection=collection,
        analyses=analyses,
        set_audit=set_audit,
        chunk_count=len(chunks),
    )

    _print_summary(
        output_path=output_path,
        extraction_path=extraction_path if save_json else None,
        analysis_path=analysis_path if save_json else None,
        collection=collection,
        analyses=analyses,
        set_audit=set_audit,
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
    set_audit: RequirementSetAudit,
) -> None:
    """Print a short execution summary."""
    status_counts = analyses.by_status()
    typer.secho("Готово.", bold=True)
    typer.echo(f"  Требований: {len(collection)}")
    for status in RequirementStatus:
        typer.echo(f"  {status.value}: {status_counts.get(status, 0)}")
    typer.echo(f"  Находок: {analyses.finding_count}")
    typer.echo(f"  Находок уровня набора: {len(set_audit.findings)}")
    typer.secho(
        f"  Вердикт: {set_audit.verdict.value}", fg=_verdict_color(set_audit.verdict)
    )
    typer.echo(f"  Отчёт: {output_path}")
    if extraction_path is not None:
        typer.echo(f"  JSON извлечения: {extraction_path}")
    if analysis_path is not None:
        typer.echo(f"  JSON анализа: {analysis_path}")


def _verdict_color(verdict: VerdictStatus) -> str:
    """Console color of a verdict."""
    if verdict is VerdictStatus.BLOCKED:
        return typer.colors.RED
    if verdict is VerdictStatus.NEEDS_REVISION:
        return typer.colors.YELLOW
    return typer.colors.GREEN


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
        "commit": get_git_commit(),
        "extractor_model": settings.extractor_model,
        "auditor_model": settings.auditor_model,
        "extractor_effort": settings.extractor_effort,
        "auditor_effort": settings.auditor_effort,
        "extractor_max_tokens": settings.extractor_max_tokens,
        "auditor_max_tokens": settings.auditor_max_tokens,
        "llm_base_url": settings.llm_base_url,
        "chunks": chunk_count,
    }


def get_git_commit() -> str:
    """Return the short hash of the agent's commit, or ``unknown`` outside git.

    The hash carries a ``-dirty`` suffix while the agent code differs from
    the commit, so a report never names a commit that cannot reproduce it.
    """
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        changes = _git("status", "--porcelain", "--", ".")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if changes else commit


def _git(*args: str) -> str:
    """Run a read-only git command in the agent's directory."""
    return subprocess.check_output(
        ["git", *args],
        cwd=_AGENT_DIR,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


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
    passport: dict[str, object],
    collection: RequirementCollection,
    analyses: RequirementAnalysisCollection,
    set_audit: RequirementSetAudit,
    chunk_count: int,
) -> None:
    """Write the Markdown report with both analysis levels and the verdict."""
    summary = _summary(analyses)
    lines: list[str] = [
        "# Отчёт проверки требований (LLM-анализ)",
        "",
        "> Отчёт сформирован автоматически по результатам реального вызова LLM.",
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
        f"| Коммит агента | {_cell(str(passport['commit']))} |",
        f"| Модель извлечения | {_cell(str(passport['extractor_model'] or '—'))} |",
        f"| Модель аудита | {_cell(str(passport['auditor_model'] or '—'))} |",
        f"| Режим извлечения | effort={_cell(str(passport['extractor_effort'] or '—'))} |",
        f"| Режим аудита | effort={_cell(str(passport['auditor_effort'] or '—'))} |",
        f"| Бюджет извлечения | {passport['extractor_max_tokens']} токенов |",
        f"| Бюджет аудита | {passport['auditor_max_tokens']} токенов |",
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

    lines.extend(_set_audit_lines(set_audit, collection))

    lines.extend(
        [
            "## Сводка",
            "",
            f"- Требований: {summary['requirements']}.",
            f"- Находок: {summary['findings']}.",
            f"- Находок уровня набора: {len(set_audit.findings)}.",
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
    blocking = [
        finding
        for finding in set_audit.findings
        if finding.severity in {Severity.HIGH, Severity.CRITICAL}
    ]
    if critical or blocking:
        lines.extend(["## Критичные и высокие дефекты", ""])
        for requirement_id, finding in critical:
            lines.append(
                f"- **{requirement_id}** ({finding.severity.value}, "
                f"{finding.criterion.value}): {finding.explanation}"
            )
        for finding in blocking:
            lines.append(
                f"- **Набор требований** ({finding.severity.value}, "
                f"{finding.category.value}): {finding.title}"
            )
        lines.append("")

    lines.extend(_verdict_lines(set_audit))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _set_audit_lines(
    set_audit: RequirementSetAudit, collection: RequirementCollection
) -> list[str]:
    """Render the set-level audit: summary, goal coverage and its findings."""
    lines = [
        "## Проверка всего набора требований",
        "",
        set_audit.summary,
        "",
        f"**Покрытие целей.** {set_audit.goal_coverage}",
        "",
    ]
    if not set_audit.findings:
        lines.extend(["Дефектов уровня всего набора не найдено.", ""])

    original_ids = {
        requirement.id: requirement.original_id for requirement in collection.requirements
    }
    for number, finding in enumerate(set_audit.findings, start=1):
        lines.extend(_set_finding_lines(number, finding, original_ids))

    if set_audit.excluded_findings:
        lines.extend(
            [
                f"> Исключено находок без подтверждённых цитат: "
                f"{set_audit.excluded_findings}. Модель сообщила о них, но дословные",
                "> цитаты в документе не найдены, поэтому они не показаны как нарушения",
                "> и требуют ручной проверки.",
                "",
            ]
        )
    return lines


def _set_finding_lines(
    number: int, finding: SetFinding, original_ids: dict[str, str | None]
) -> list[str]:
    """Render one set-level finding together with its verified quotes."""
    related = ", ".join(
        requirement_id if not original_ids.get(requirement_id)
        else f"{requirement_id} ({original_ids[requirement_id]})"
        for requirement_id in finding.related_req_ids
    )
    lines = [
        f"### {number}. {finding.title}",
        "",
        f"- Категория: {finding.category.value}",
        f"- Severity: {finding.severity.value}",
        f"- Связанные требования: {related or '—'}",
        f"- Описание: {finding.description}",
        "- Доказательство:",
    ]
    for quote in finding.evidence:
        lines.append(f"  - «{_cell(quote)}»")
    lines.append("")
    return lines


def _verdict_lines(set_audit: RequirementSetAudit) -> list[str]:
    """Render the final verdict and what to fix first."""
    lines = [
        "## Итог",
        "",
        f"**Вердикт: {set_audit.verdict.value}.** {set_audit.verdict_reason}",
        "",
    ]
    if set_audit.priority_actions:
        lines.extend(["Что исправить в первую очередь:", ""])
        for number, action in enumerate(set_audit.priority_actions, start=1):
            lines.append(f"{number}. {_cell(action)}")
        lines.append("")
    return lines


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
