"""Чтение образца ответа и проверка итогового CSV"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def read_sample(path: str | Path) -> pd.DataFrame:
    """Прочитать образец и проверить состав тестовых ID"""

    path = Path(path)
    with path.open("rb") as file:
        if file.read(100).startswith(b"version https://git-lfs"):
            raise ValueError("Образец является указателем Git LFS")

    sample = pd.read_csv(path, dtype={"image_id": str})
    if list(sample.columns) != ["image_id", "p_180"]:
        raise ValueError("Ожидались колонки image_id и p_180")

    if len(sample) != 20_000:
        raise ValueError("Ожидалось 20 000 строк в образце")

    ids = sample["image_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("В образце есть пустые или повторяющиеся ID")

    expected = {f"test_{index:05d}" for index in range(20_000)}
    if set(ids) != expected:
        raise ValueError("ID отличаются от test_00000 ... test_19999")

    return sample


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    """Проверить колонки, порядок ID и диапазон вероятностей"""

    if list(submission.columns) != ["image_id", "p_180"]:
        raise ValueError("Ожидались только колонки image_id и p_180")

    ids = submission["image_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("В ответе есть пустые или повторяющиеся ID")

    # Сохраняем порядок образца: перестановка строк может сопоставить
    # правильные вероятности с чужими изображениями при проверке ответа.
    if ids.tolist() != sample["image_id"].tolist():
        raise ValueError("Количество или порядок ID не совпадает с образцом")

    probabilities = submission["p_180"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("В вероятностях есть пропуски или бесконечности")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Вероятности должны находиться в диапазоне [0, 1]")


def save_submission(
    sample: pd.DataFrame,
    probabilities: NDArray[np.float64],
    path: str | Path,
) -> pd.DataFrame:
    """Сохранить ответ после проверки записанного CSV"""

    if probabilities.shape != (len(sample),):
        raise ValueError("Число вероятностей не совпадает с числом ID")

    submission = sample.copy()
    submission["p_180"] = probabilities
    validate_submission(submission, sample)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")

    try:
        # Оставляем ту же точность записи, с которой получен отправленный CSV.
        # Проверяем уже записанные числа, а не только массив до сериализации.
        submission.to_csv(temporary, index=False, float_format="%.10f")
        saved = pd.read_csv(temporary, dtype={"image_id": str})
        validate_submission(saved, sample)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return saved


def compare_submissions(
    candidate_path: str | Path,
    reference_path: str | Path,
    sample: pd.DataFrame,
) -> dict[str, bool | int | float]:
    """Сравнить новый CSV с отправленным без изменения вероятностей

    Сначала проверяем оба файла относительно одного образца. Это исключает
    сравнение вероятностей разных изображений из-за перестановки строк.
    Побайтовое совпадение и разница чисел проверяются отдельно: файлы могут
    отличаться, например, переводами строк при одинаковых вероятностях.
    Тестовых меток здесь нет, поэтому качество модели функция не оценивает.
    """

    candidate_path, reference_path = Path(candidate_path), Path(reference_path)
    candidate = pd.read_csv(
        candidate_path, dtype={"image_id": str}, float_precision="round_trip"
    )
    reference = pd.read_csv(
        reference_path, dtype={"image_id": str}, float_precision="round_trip"
    )
    validate_submission(candidate, sample)
    validate_submission(reference, sample)

    difference = np.abs(
        candidate["p_180"].to_numpy(dtype=np.float64)
        - reference["p_180"].to_numpy(dtype=np.float64)
    )
    with candidate_path.open("rb") as first, reference_path.open("rb") as second:
        same_bytes = (
            hashlib.file_digest(first, "sha256").digest()
            == hashlib.file_digest(second, "sha256").digest()
        )

    return {
        "rows": len(candidate),
        "identical_bytes": same_bytes,
        "different_probabilities": int(np.count_nonzero(difference)),
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
    }
