"""Unit tests for entity-based requirement clustering."""

from __future__ import annotations

from app.models.requirement import Requirement
from app.services.requirement_clustering import (
    batch_context_entities,
    build_requirement_batches,
    extract_entities,
)


def _requirements(count: int, *, related_at: tuple[int, int] | None = None):
    items = [
        Requirement(
            id=f"REQ-{number:03d}",
            text=f"Уникальное правило номер {number} про редкость{number}.",
        )
        for number in range(1, count + 1)
    ]
    if related_at is not None:
        first, second = related_at
        items[first - 1] = Requirement(
            id=f"REQ-{first:03d}",
            text="Клиент задаёт порог 50 000 тиын через /alerts/saveLimit.",
        )
        items[second - 1] = Requirement(
            id=f"REQ-{second:03d}",
            text="Сервис сохраняет порог 50 000 тиын через /alerts/saveLimit.",
        )
    return items


class TestBatching:
    def test_all_requirements_appear_once_and_batches_are_capped(self) -> None:
        requirements = _requirements(45)

        batches = build_requirement_batches(requirements, max_batch_size=20)

        ids = [item.id for batch in batches for item in batch]
        assert sorted(ids) == sorted(item.id for item in requirements)
        assert all(len(batch) <= 20 for batch in batches)
        assert len(batches) == 3

    def test_related_endpoints_stay_in_the_same_batch(self) -> None:
        requirements = _requirements(25, related_at=(1, 25))

        batches = build_requirement_batches(requirements, max_batch_size=20)

        batch_with_first = next(
            batch for batch in batches if batch[0].id == "REQ-001"
        )
        assert "REQ-025" in [item.id for item in batch_with_first]

    def test_small_collection_is_one_batch(self) -> None:
        requirements = _requirements(3)

        assert build_requirement_batches(requirements, 20) == [requirements]

    def test_empty_collection_has_no_batches(self) -> None:
        assert build_requirement_batches([], 20) == []


class TestEntities:
    def test_endpoint_identifier_amount_and_unit_are_extracted(self) -> None:
        entities = extract_entities(
            "Клиент задаёт порог 50 000 тиын через /alerts/saveLimit."
        )

        assert "endpoint:/alerts/savelimit" in entities
        assert "endpoint_segment:savelimit" in entities
        assert "id:savelimit" in entities
        assert "unit:tiyn" in entities
        assert "value:50000:tiyn" in entities

    def test_shared_context_is_readable(self) -> None:
        batch = [
            Requirement(
                id="REQ-001",
                text="Порог 50 000 тиын хранится в поле amount и меняется через /alerts/saveLimit.",
            ),
            Requirement(
                id="REQ-002",
                text="Порог 50 000 тиын меняется через /alerts/saveLimit.",
            ),
        ]

        context = batch_context_entities(batch)

        assert "tiyn" in context
        assert "50000:tiyn" in context
        assert "/alerts/savelimit" in context
