import numpy as np
import pytest

from metrics import average_precision_at_10, mean_average_precision_at_10


def test_average_precision_uses_relevant_count() -> None:
    """Проверяет знаменатель по числу релевантных документов"""
    ranking = np.array([1, 2, 3, 10, 11, 12, 13, 14, 15, 16])

    assert average_precision_at_10(ranking, {1, 2, 3, 4}) == pytest.approx(0.75)


def test_average_precision_caps_denominator_at_10() -> None:
    """Проверяет ограничение знаменателя десятью документами"""
    ranking = np.arange(1, 11)

    assert average_precision_at_10(ranking, set(range(1, 13))) == pytest.approx(1.0)


def test_average_precision_ignores_positions_after_10() -> None:
    """Проверяет отсечение документов после десятой позиции"""
    ranking = np.arange(1, 12)

    assert average_precision_at_10(ranking, {11}) == pytest.approx(0.0)


def test_average_precision_respects_order() -> None:
    """Проверяет влияние позиции релевантных документов"""
    ranking = np.array([3, 1, 2, 4])

    assert average_precision_at_10(ranking, {1, 2}) == pytest.approx(7 / 12)


def test_average_precision_ignores_duplicate_hits() -> None:
    """Проверяет отсутствие повторного учёта одной статьи"""
    ranking = np.array([1, 1, 2, 3])

    assert average_precision_at_10(ranking, {1, 2}) == pytest.approx(5 / 6)


def test_mean_average_precision_averages_queries() -> None:
    """Проверяет усреднение AP по нескольким запросам"""
    rankings = np.array([[1, 2, 3], [3, 2, 1]])

    assert mean_average_precision_at_10(rankings, [{1}, {1}]) == pytest.approx(2 / 3)
