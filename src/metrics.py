from collections.abc import Collection, Sequence

import numpy as np

import config


def average_precision_at_10(
    ranking: Sequence[int] | np.ndarray,
    relevant: Collection[int],
) -> float:
    """Вычисляет AP@10 для одного ранжированного списка"""
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0

    seen_ids: set[int] = set()
    hits = 0
    score = 0.0

    for rank, value in enumerate(ranking[: config.METRIC_CUTOFF], start=1):
        article_id = int(value)

        if article_id in relevant_ids and article_id not in seen_ids:
            hits += 1
            score += hits / rank

        seen_ids.add(article_id)

    denominator = min(len(relevant_ids), config.METRIC_CUTOFF)

    return score / denominator


def mean_average_precision_at_10(
    rankings: np.ndarray,
    ground_truth: Sequence[Collection[int]],
) -> float:
    """Вычисляет среднее значение AP@10 по всем запросам"""
    scores = [
        average_precision_at_10(ranking, relevant)
        for ranking, relevant in zip(rankings, ground_truth, strict=True)
    ]

    return float(np.mean(scores)) if scores else 0.0
