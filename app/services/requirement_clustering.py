"""Entity-based clustering of requirements into auditor batches.

Blindly slicing requirements by number is cheap but loses context: a rule
about one push per month can land in batch 1 while the endpoint that stores
the limit lands in batch 2. This module builds a tiny entity graph (endpoints,
HTTP codes, field names, amounts, units, domain terms and rare words) and
keeps related requirements in the same batch. Batches still respect the
model's output-token limit (20 requirements by default).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from app.models.requirement import Requirement

_ENDPOINT_RE = re.compile(r"(?<![\w])/[a-zA-Z0-9_\-/{}.]+")
_HTTP_CODE_RE = re.compile(r"\b[1-5]\d{2}\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", re.IGNORECASE)
_CAMEL_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b")
_QUOTED_RE = re.compile(r"[«\"`]([^»\"`]{2,60})[»\"`]")
_NUMBER_UNIT_RE = re.compile(
    r"\b(\d[\d\s.,]*)\s*"
    r"(₸|тенге|тиын|минут[а-яё]*|час[а-яё]*|день|дня|дней|месяц[а-яё]*"
    r"|год[а-яё]*|%|процент[а-яё]*)(?![а-яёa-z])",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[а-яёa-z]{4,}", re.IGNORECASE)

_STRONG_PREFIXES = (
    "endpoint:",
    "endpoint_segment:",
    "http:",
    "id:",
    "term:",
    "value:",
)
_UNIT_PREFIX = "unit:"

_DOMAIN_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("push", re.compile(r"\bpush\b|пуш", re.IGNORECASE)),
    ("sms", re.compile(r"\bsms\b|смс", re.IGNORECASE)),
    ("email", re.compile(r"\bemail\b|почт", re.IGNORECASE)),
    ("api", re.compile(r"\bapi\b|\brest\b|эндпоинт|endpoint", re.IGNORECASE)),
    ("card", re.compile(r"карт", re.IGNORECASE)),
    ("limit", re.compile(r"порог|лимит|threshold|\blimit\b", re.IGNORECASE)),
    ("expense", re.compile(r"расход|трат", re.IGNORECASE)),
    ("purchase", re.compile(r"покупк|транзакц|операц", re.IGNORECASE)),
    ("refund", re.compile(r"возврат", re.IGNORECASE)),
    ("transfer", re.compile(r"перевод", re.IGNORECASE)),
    ("cash", re.compile(r"наличн", re.IGNORECASE)),
    ("currency", re.compile(r"тенге|тиын|₸|валют|\bkzt\b", re.IGNORECASE)),
    ("month", re.compile(r"месяц|календарн", re.IGNORECASE)),
    ("notification", re.compile(r"уведомлен|оповещен", re.IGNORECASE)),
    ("settings", re.compile(r"настройк|конфигурац", re.IGNORECASE)),
    ("storage", re.compile(r"хранилищ|хранен", re.IGNORECASE)),
    (
        "auth",
        re.compile(r"авторизац|аутентификац|bearer|токен|сесси", re.IGNORECASE),
    ),
    ("error", re.compile(r"ошибк|\berror\b", re.IGNORECASE)),
    ("status", re.compile(r"статус|\bstatus\b", re.IGNORECASE)),
    ("enabled", re.compile(r"включ|переключател|\benabled\b", re.IGNORECASE)),
    ("almaty", re.compile(r"алматы", re.IGNORECASE)),
    ("processing", re.compile(r"процессинг", re.IGNORECASE)),
    ("test", re.compile(r"тестирован|\bтест", re.IGNORECASE)),
    ("review", re.compile(r"ревью|проверк", re.IGNORECASE)),
)

_STOPWORDS = frozenset(
    {
        "система",
        "системы",
        "системе",
        "систему",
        "должен",
        "должна",
        "должно",
        "должны",
        "требование",
        "требования",
        "требований",
        "документ",
        "документа",
        "возможность",
        "обеспечить",
        "обеспечивает",
        "является",
        "являются",
        "который",
        "которая",
        "которое",
        "которые",
        "которых",
        "которым",
        "которую",
        "этот",
        "эта",
        "эти",
        "этого",
        "этом",
        "все",
        "весь",
        "вся",
        "свой",
        "свои",
        "своих",
        "также",
        "после",
        "перед",
        "между",
        "через",
        "более",
        "менее",
        "только",
        "нужно",
        "необходимо",
        "можно",
        "быть",
        "был",
        "была",
        "было",
        "были",
        "есть",
        "нет",
        "или",
        "если",
        "как",
        "что",
        "чтобы",
        "при",
        "для",
        "без",
        "над",
        "под",
        "из",
        "от",
        "до",
        "за",
        "по",
        "на",
        "в",
        "во",
        "с",
        "со",
        "у",
        "о",
        "об",
        "же",
        "ли",
        "бы",
        "то",
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "must",
        "should",
        "shall",
        "will",
        "can",
        "system",
        "requirement",
        "requirements",
        "document",
    }
)


def build_requirement_batches(
    requirements: Sequence[Requirement], max_batch_size: int = 20
) -> list[list[Requirement]]:
    """Group requirements into batches of at most ``max_batch_size``.

    Related requirements (shared endpoints, HTTP codes, identifiers, amounts
    or rare entities) stay together. Small clusters are packed into shared
    batches; a cluster is never split unless it is larger than the limit.
    """
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    items = list(requirements)
    if not items:
        return []
    if len(items) <= max_batch_size:
        return [items]

    entity_sets = {index: extract_entities(item.text) for index, item in enumerate(items)}
    idf = _inverse_document_frequency(entity_sets)
    similarities = _similarity_table(entity_sets, idf)
    clusters = _connected_components(items, entity_sets, similarities)

    split: list[list[int]] = []
    for cluster in clusters:
        if len(cluster) <= max_batch_size:
            split.append(cluster)
        else:
            split.extend(
                _split_large_cluster(cluster, similarities, max_batch_size)
            )

    packed = _pack_clusters(split, max_batch_size)
    packed = _merge_batches(packed, max_batch_size)
    packed.sort(key=min)
    return [
        [items[index] for index in sorted(batch)]
        for batch in packed
    ]


def extract_entities(text: str) -> set[str]:
    """Extract the lightweight entities used for clustering."""
    entities: set[str] = set()
    for match in _ENDPOINT_RE.findall(text):
        endpoint = match.casefold().rstrip(".,;:)")
        if not endpoint:
            continue
        entities.add(f"endpoint:{endpoint}")
        segment = endpoint.rstrip("/").rsplit("/", 1)[-1]
        if segment:
            entities.add(f"endpoint_segment:{segment}")

    for code in _HTTP_CODE_RE.findall(text):
        entities.add(f"http:{code}")

    for match in [*_SNAKE_RE.findall(text), *_CAMEL_RE.findall(text)]:
        entities.add(f"id:{match.casefold()}")

    for match in _QUOTED_RE.findall(text):
        normalized = " ".join(match.casefold().split())
        if normalized:
            entities.add(f"term:{normalized}")

    for number, unit in _NUMBER_UNIT_RE.findall(text):
        canonical_unit = _canonical_unit(unit)
        if not canonical_unit:
            continue
        entities.add(f"{_UNIT_PREFIX}{canonical_unit}")
        entities.add(f"value:{_normalize_number(number)}:{canonical_unit}")

    for name, pattern in _DOMAIN_TERMS:
        if pattern.search(text):
            entities.add(f"domain:{name}")

    for word in _WORD_RE.findall(text.casefold()):
        if word in _STOPWORDS:
            continue
        if re.match(r"[а-яё]", word):
            stem = word[:4]
        else:
            stem = word[:-1] if word.endswith("s") and len(word) > 4 else word
        if len(stem) >= 4:
            entities.add(f"word:{stem}")

    return entities


def batch_context_entities(batch: Sequence[Requirement]) -> list[str]:
    """Return readable entities shared by at least two requirements in a batch."""
    counts: Counter[str] = Counter()
    for item in batch:
        counts.update(
            entity
            for entity in extract_entities(item.text)
            if entity.startswith(_STRONG_PREFIXES)
            or entity.startswith(_UNIT_PREFIX)
            or entity.startswith("domain:")
        )
    shared = [
        (entity, count)
        for entity, count in counts.items()
        if count >= 2
    ]
    shared.sort(key=lambda item: (-item[1], item[0]))
    return [entity.split(":", 1)[-1] for entity, _ in shared[:10]]


def _inverse_document_frequency(
    entity_sets: dict[int, set[str]]
) -> dict[str, float]:
    total = len(entity_sets)
    document_frequency: Counter[str] = Counter()
    for entities in entity_sets.values():
        document_frequency.update(entities)
    return {
        entity: math.log(1 + total / frequency)
        for entity, frequency in document_frequency.items()
    }


def _similarity_table(
    entity_sets: dict[int, set[str]], idf: dict[str, float]
) -> dict[tuple[int, int], float]:
    weights = {
        index: math.sqrt(sum(idf[entity] ** 2 for entity in entities))
        for index, entities in entity_sets.items()
    }
    table: dict[tuple[int, int], float] = {}
    indices = sorted(entity_sets)
    for position, left in enumerate(indices):
        for right in indices[position + 1 :]:
            shared = entity_sets[left] & entity_sets[right]
            if not shared:
                table[(left, right)] = 0.0
                continue
            numerator = sum(idf[entity] for entity in shared)
            denominator = weights[left] * weights[right]
            table[(left, right)] = numerator / denominator if denominator else 0.0
    return table


def _similarity(
    similarities: dict[tuple[int, int], float], left: int, right: int
) -> float:
    if left == right:
        return 1.0
    return similarities.get((min(left, right), max(left, right)), 0.0)


def _connected_components(
    items: Sequence[Requirement],
    entity_sets: dict[int, set[str]],
    similarities: dict[tuple[int, int], float],
) -> list[list[int]]:
    total = len(items)
    parent = list(range(total))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(total):
        for right in range(left + 1, total):
            if _should_connect(
                entity_sets[left],
                entity_sets[right],
                similarities[(left, right)],
            ):
                union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(total):
        grouped[find(index)].append(index)
    return [grouped[root] for root in sorted(grouped)]


def _should_connect(
    left_entities: set[str],
    right_entities: set[str],
    similarity: float,
) -> bool:
    shared = left_entities & right_entities
    if any(entity.startswith(_STRONG_PREFIXES) for entity in shared):
        return True
    return similarity >= 0.20


def _split_large_cluster(
    cluster: list[int],
    similarities: dict[tuple[int, int], float],
    max_batch_size: int,
) -> list[list[int]]:
    remaining = set(cluster)
    batches: list[list[int]] = []
    while remaining:
        seed = max(
            remaining,
            key=lambda index: (
                sum(
                    _similarity(similarities, index, other)
                    for other in remaining
                    if other != index
                ),
                -index,
            ),
        )
        batch = [seed]
        remaining.remove(seed)
        while remaining and len(batch) < max_batch_size:
            candidate = max(
                remaining,
                key=lambda index: (
                    sum(_similarity(similarities, index, chosen) for chosen in batch),
                    -index,
                ),
            )
            batch.append(candidate)
            remaining.remove(candidate)
        batches.append(batch)
    return batches


def _pack_clusters(
    clusters: list[list[int]], max_batch_size: int
) -> list[list[int]]:
    ordered = sorted(clusters, key=lambda cluster: (-len(cluster), min(cluster)))
    batches: list[list[int]] = []
    for cluster in ordered:
        for batch in batches:
            if len(batch) + len(cluster) <= max_batch_size:
                batch.extend(cluster)
                break
        else:
            batches.append(list(cluster))
    return batches


def _merge_batches(
    batches: list[list[int]], max_batch_size: int
) -> list[list[int]]:
    changed = True
    while changed:
        changed = False
        for left in range(len(batches)):
            for right in range(left + 1, len(batches)):
                if len(batches[left]) + len(batches[right]) <= max_batch_size:
                    batches[left].extend(batches[right])
                    del batches[right]
                    changed = True
                    break
            if changed:
                break
    return batches


def _canonical_unit(unit: str) -> str | None:
    normalized = unit.casefold()
    if normalized in {"₸", "тенге"}:
        return "kzt"
    if normalized.startswith("тиын"):
        return "tiyn"
    if normalized.startswith("минут"):
        return "minute"
    if normalized.startswith("час"):
        return "hour"
    if normalized.startswith(("день", "дня", "дней")):
        return "day"
    if normalized.startswith("месяц"):
        return "month"
    if normalized.startswith("год"):
        return "year"
    if normalized in {"%", "процент"} or normalized.startswith("процент"):
        return "percent"
    return None


def _normalize_number(value: str) -> str:
    return value.replace(" ", "").replace(",", ".").strip(".")
