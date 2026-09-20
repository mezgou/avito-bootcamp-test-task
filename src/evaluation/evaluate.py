"""Оценка сохранённых весов на исходных и перевёрнутых текстовых кропах"""

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import paddle
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from src.data.prepare import read_manifest
from src.data.preprocessing import preprocess_image, read_rgb
from src.data.sources import save_json
from src.evaluation.calibrate import VARIANTS, apply_temperature
from src.evaluation.metrics import summarize_metrics
from src.models.lcnet import PPLCNet
from src.models.orientation import load_model, predict_batch

PREPROCESSING = "RGB, rotation before resize, 160x80, ImageNet, FP32"


def _prepare_pair(path: str | Path) -> NDArray[np.float32]:
    """Подготовить исходный кроп и его поворот до изменения размера"""

    image = read_rgb(path)
    rotated = np.ascontiguousarray(image[::-1, ::-1])

    return np.stack((preprocess_image(image), preprocess_image(rotated)))


def predict_pairs(
    model: PPLCNet,
    paths: Sequence[str | Path],
    batch_size: int = 256,
    read_workers: int = 8,
) -> NDArray[np.float64]:
    """Получить p_180 для двух ориентаций каждого кропа в порядке paths

    Результат имеет форму (число кропов, 2): исходный кроп, затем поворот.
    batch_size ограничивает число изображений в одном вызове модели.
    Пара не разделяется между пачками; нечётный размер округляется вниз.
    """

    if batch_size < 2 or read_workers < 1:
        raise ValueError("Нужны batch_size >= 2 и read_workers >= 1")

    if len(paths) == 0:
        raise ValueError("Список кропов пуст")

    cv2.setNumThreads(1)
    crops_per_batch = batch_size // 2
    probabilities = np.empty((len(paths), 2), dtype=np.float64)
    starts = range(0, len(paths), crops_per_batch)

    with ThreadPoolExecutor(max_workers=read_workers) as pool:
        for start in tqdm(starts, desc="Оценка двух ориентаций", unit="batch"):
            stop = min(start + crops_per_batch, len(paths))
            pairs = list(pool.map(_prepare_pair, paths[start:stop]))
            batch = np.concatenate(pairs, axis=0)
            values = predict_batch(model, batch)

            if (
                values.shape != (2 * (stop - start),)
                or not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()
            ):
                raise ValueError("Модель вернула некорректные вероятности")

            probabilities[start:stop] = values.reshape(-1, 2)

    return probabilities


def build_predictions(
    frame: pd.DataFrame,
    probabilities: NDArray[np.float64],
) -> pd.DataFrame:
    """Собрать прогнозы single и rotation с метками обеих ориентаций

    Все PNG из manifest уже ориентированы нормально: исходный кроп имеет
    метку 0, созданный здесь поворот — метку 1. applied_rotation описывает
    подготовку исходных данных и не используется как метка класса.

    single: p(x).
    rotation: (p(x) + 1 - p(rot180(x))) / 2.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)

    if frame.empty or probabilities.shape != (len(frame), 2):
        raise ValueError("Для каждого кропа нужны вероятности двух ориентаций")

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Вероятности должны быть конечными числами от 0 до 1")

    columns = [
        "crop_id", "path", "scene_id", "group_id", "split", "source",
        "dataset", "kind", "text", "script", "width", "height",
    ]
    repeated = np.repeat(np.arange(len(frame)), 2)
    base = frame.iloc[repeated][columns].reset_index(drop=True)
    base["target"] = np.tile([0, 1], len(frame))

    single = base.assign(variant="single", p_180=probabilities.ravel())
    upright = 0.5 * (probabilities[:, 0] + (1.0 - probabilities[:, 1]))
    combined = np.column_stack((upright, 1.0 - upright))
    rotation = base.assign(variant="rotation", p_180=combined.ravel())

    return pd.concat((single, rotation), ignore_index=True)


def evaluate_model(
    model: PPLCNet,
    frame: pd.DataFrame,
    root: Path,
    batch_size: int = 256,
    read_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оценить модель на одной части проверенного manifest

    root — каталог с manifest.csv; пути кропов считаются относительно него.
    Возвращаются подробные прогнозы и метрики по источникам и режимам.
    Модель остаётся в eval; при продолжении обучения нужен model.train().
    """

    if frame.empty or frame.split.nunique() != 1:
        raise ValueError("Передайте одну непустую часть данных")

    paths = [root / value for value in frame.path]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Не найден кроп: {path}")

    probabilities = predict_pairs(model, paths, batch_size, read_workers)
    predictions = build_predictions(frame, probabilities)
    summary = summarize_metrics(predictions)

    return predictions, summary


def _sha256(path: Path) -> str:
    """Зафиксировать содержимое весов или manifest в отчёте"""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Записать CSV через временный файл, сохранив выбранное сжатие"""

    temporary = path.with_name(path.name + ".part")
    compression = "gzip" if path.suffix == ".gz" else None

    try:
        frame.to_csv(temporary, index=False, compression=compression)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_calibration(path: Path, weights_hash: str, manifest_hash: str) -> dict:
    """Проверить связь готовой калибровки с весами и разделением данных"""

    settings = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(settings, dict):
        raise ValueError("Ожидался JSON с настройками калибровки")

    expected = {
        "method": "temperature",
        "apply_to": "variant_probability",
        "formula": "sigmoid(logit(p) / temperature)",
        "fit_split": "calibration",
        "fit_source": "real",
        "criterion": "brier",
        "weights_sha256": weights_hash,
        "manifest_sha256": manifest_hash,
        "preprocessing": PREPROCESSING,
    }

    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Калибровка не соответствует текущему запуску: {key}")

    variant = settings.get("variant")

    if variant not in VARIANTS or settings.get("variant_formula") != VARIANTS[variant]:
        raise ValueError("Не совпадает формула режима предсказания")

    temperature = settings.get("temperature")

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("В настройках указана некорректная температура")

    return settings


def main() -> None:
    """Оценить checkpoint, при необходимости применив готовую калибровку"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/external/manifest.csv")
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--split",
        choices=("development", "calibration", "holdout"),
        default="development",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--read-workers", type=int, default=8)
    parser.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    args = parser.parse_args()

    if args.batch_size < 2 or args.read_workers < 1:
        parser.error("Нужны --batch-size >= 2 и --read-workers >= 1")

    for path in (args.manifest, args.weights):
        if not path.is_file():
            parser.error(f"Не найден файл: {path}")

    # Отдельный каталог сохраняет связь прогнозов с конкретными весами.
    # Незавершённый запуск тоже остаётся доступен для разбора.
    if args.output.exists():
        parser.error("Каталог результатов уже существует; укажите новый --output")

    frame = read_manifest(args.manifest)
    frame = frame.loc[frame.split == args.split].reset_index(drop=True)

    if frame.empty:
        parser.error(f"В manifest нет кропов для части {args.split}")

    weights_hash = _sha256(args.weights)
    manifest_hash = _sha256(args.manifest)
    calibration = None
    temperature = 1.0
    variants = VARIANTS.copy()

    if args.calibration is not None:
        calibration = _read_calibration(args.calibration, weights_hash, manifest_hash)
        temperature = float(calibration["temperature"])
        variant = calibration["variant"]
        variants = {variant: VARIANTS[variant]}

    started = time.perf_counter()
    model = load_model(args.weights, device=args.device)
    device = paddle.device.get_device()

    print(f"Устройство: {device}; веса: {args.weights}", flush=True)
    print(
        f"Часть: {args.split}; кропов: {len(frame)}; "
        f"примеров с ориентацией: {2 * len(frame)}",
        flush=True,
    )
    if calibration is None:
        print("Режимы: single, rotation; без калибровки", flush=True)
    else:
        print(
            f"Режим: {calibration['variant']}; фиксированная температура: {temperature:.8f}",
            flush=True,
        )

    predictions, summary = evaluate_model(
        model,
        frame,
        args.manifest.parent,
        batch_size=args.batch_size,
        read_workers=args.read_workers,
    )

    if calibration is not None:
        predictions = predictions.loc[
            predictions.variant == calibration["variant"]
        ].copy().reset_index(drop=True)
        predictions["p_180_raw"] = predictions.p_180
        predictions["p_180"] = apply_temperature(
            predictions.p_180.to_numpy(), temperature
        )
        summary = summarize_metrics(predictions)

    report = {
        "weights": str(args.weights.resolve()),
        "weights_sha256": weights_hash,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "split": args.split,
        "device": device,
        "batch_size": args.batch_size,
        "read_workers": args.read_workers,
        "crops": len(frame),
        "examples_per_variant": 2 * len(frame),
        "labels": {"0": "upright", "1": "rotated_180"},
        "variants": variants,
        "preprocessing": PREPROCESSING,
        "temperature": temperature,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "metrics": summary.to_dict(orient="records"),
    }

    if calibration is not None:
        report["calibration"] = calibration
        report["calibration_path"] = str(args.calibration.resolve())
        report["temperature_stage"] = "after_variant_probability"

    args.output.mkdir(parents=True, exist_ok=False)
    _save_csv(predictions, args.output / "predictions.csv.gz")
    _save_csv(summary, args.output / "metrics.csv")
    save_json(args.output / "report.json", report)

    print(summary.to_string(index=False, float_format="%.8f"), flush=True)
    print(f"Результаты: {args.output}", flush=True)


if __name__ == "__main__":
    main()
    