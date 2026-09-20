"""Метрики вероятности поворота текста на 180 градусов"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# Ограничение применяется только при вычислении логарифма.
LOG_EPSILON = 1e-15


def binary_metrics(
    targets: ArrayLike,
    probabilities: ArrayLike,
) -> dict[str, int | float]:
    """Посчитать Brier, score, log loss и accuracy по вероятностям класса 1

    targets и probabilities — непустые одномерные массивы одинаковой длины.
    Метка 0 означает нормальный текст, метка 1 — поворот на 180 градусов.
    Порядок вероятностей должен совпадать с порядком меток.

    Brier считается по исходным вероятностям; score равен 1 - Brier.
    Для log loss вероятность истинного класса ограничивается снизу 1e-15.
    При вычислении accuracy вероятность >= 0.5 соответствует классу 1.
    """

    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)

    if targets.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Метки и вероятности должны быть одномерными массивами")

    if targets.size == 0 or targets.shape != probabilities.shape:
        raise ValueError("Нужны непустые массивы одинаковой длины")

    if targets.dtype.kind not in "biuf" or probabilities.dtype.kind not in "biuf":
        raise ValueError("Метки и вероятности должны быть числовыми")

    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Метки должны принимать только значения 0 и 1")

    if not np.isfinite(probabilities).all():
        raise ValueError("Вероятности содержат NaN или бесконечность")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Вероятности должны находиться в диапазоне [0, 1]")

    targets = targets.astype(np.float64, copy=False)
    probabilities = probabilities.astype(np.float64, copy=False)

    errors = probabilities - targets
    brier = float(np.mean(errors * errors))
    true_probability = np.where(targets == 1, probabilities, 1 - probabilities)
    log_loss = float(np.mean(-np.log(np.maximum(true_probability, LOG_EPSILON))))
    accuracy = float(np.mean((probabilities >= 0.5) == targets))

    return {
        "n": int(targets.size),
        "brier": brier,
        "score": 1.0 - brier,
        "log_loss": log_loss,
        "accuracy": accuracy,
    }


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Собрать метрики отдельно по частям данных и реальному/синтетическому источнику

    Одна строка predictions соответствует одной ориентации одного кропа.
    Обязательные столбцы: crop_id, split, source, target, p_180.
    Если передан столбец variant, режимы предсказания оцениваются отдельно.
    Таблица должна содержать прогнозы одного checkpoint.

    n — число примеров с ориентацией, crops — число исходных кропов.
    При оценке обеих ориентаций каждого кропа n = 2 * crops.
    Функция не смешивает real и synthetic в общую оценку.
    """

    required = {"crop_id", "split", "source", "target", "p_180"}
    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(f"В прогнозах нет столбцов: {sorted(missing)}")

    if predictions.empty:
        raise ValueError("Таблица прогнозов пуста")

    groups = ["split", "source"]

    if "variant" in predictions.columns:
        groups.append("variant")

    for column in ["crop_id", *groups]:
        valid = predictions[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )

        if not valid.all():
            raise ValueError(f"Столбец {column} должен содержать непустые строки")

    if not predictions.source.isin(("real", "synthetic")).all():
        raise ValueError("Ожидались источники real и synthetic")

    identity = [*groups, "crop_id", "target"]

    if predictions.duplicated(identity).any():
        raise ValueError("Повторяются прогнозы одной ориентации кропа в одном режиме")

    if predictions.groupby("crop_id").split.nunique().max() != 1:
        raise ValueError("Один кроп присутствует в разных частях данных")

    if predictions.groupby("crop_id").source.nunique().max() != 1:
        raise ValueError("Один кроп относится к разным источникам")

    records = []

    for values, frame in predictions.groupby(groups, sort=True, dropna=False):
        metrics = binary_metrics(
            frame.target.to_numpy(),
            frame.p_180.to_numpy(),
        )
        records.append(
            {
                **dict(zip(groups, values, strict=True)),
                "crops": int(frame.crop_id.nunique()),
                **metrics,
            }
        )

    return pd.DataFrame(records)
