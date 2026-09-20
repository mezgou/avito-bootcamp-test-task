"""Предсказание ориентации текста и сохранение сабмишена"""

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
from numpy.typing import NDArray
from tqdm.auto import tqdm

from src.data.preprocessing import preprocess_image, read_rgb
from src.data.sources import save_json
from src.evaluation.calibrate import VARIANTS, apply_temperature
from src.evaluation.evaluate import predict_pairs
from src.inference.submission import read_sample, save_submission
from src.models.lcnet import PPLCNet
from src.models.orientation import load_model, predict_batch
from src.models.weights import download_pretrained

PREPROCESSING = "RGB, rotation before resize, 160x80, ImageNet, FP32"


def _prepare_image(path: str | Path) -> NDArray[np.float32]:
    """Прочитать кроп и подготовить вход модели"""

    return preprocess_image(read_rgb(path))


def predict_paths(
    model: PPLCNet,
    paths: Sequence[str | Path],
    batch_size: int = 128,
    read_workers: int = 8,
    *,
    variant: str = "single",
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    """Получить p_180 в порядке paths с выбранным режимом и температурой

    single использует один проход. rotation усредняет p(x) и 1-p(rot180(x)),
    затем применяет температуру к полученной вероятности. Поворот выполняется
    до изменения размера тем же кодом, который используется в evaluate.
    batch_size ограничивает число изображений в одном вызове модели.
    """

    if variant not in VARIANTS:
        raise ValueError("Используйте режим single или rotation")

    minimum_batch = 2 if variant == "rotation" else 1

    if batch_size < minimum_batch or read_workers < 1:
        raise ValueError(
            f"Для {variant} нужны batch_size >= {minimum_batch} и read_workers >= 1"
        )

    if len(paths) == 0:
        raise ValueError("Список кропов пуст")

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Температура должна быть положительным конечным числом")

    if variant == "rotation":
        pairs = predict_pairs(model, paths, batch_size, read_workers)
        probabilities = 0.5 * (pairs[:, 0] + (1.0 - pairs[:, 1]))
    else:
        cv2.setNumThreads(1)
        probabilities = np.empty(len(paths), dtype=np.float64)
        starts = range(0, len(paths), batch_size)

        with ThreadPoolExecutor(max_workers=read_workers) as pool:
            for start in tqdm(starts, desc="Предсказание", unit="batch"):
                stop = min(start + batch_size, len(paths))
                batch = np.stack(list(pool.map(_prepare_image, paths[start:stop])))
                values = predict_batch(model, batch)

                if values.shape != (stop - start,):
                    raise ValueError("Число вероятностей не совпадает с размером пачки")

                probabilities[start:stop] = values

    return apply_temperature(probabilities, temperature)


def load_calibration(path: Path, weights_sha256: str) -> dict:
    """Прочитать температуру и проверить её связь с весами и обработкой входа"""

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
        "preprocessing": PREPROCESSING,
    }

    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Несовместимая настройка калибровки: {key}")

    variant = settings.get("variant")

    if variant not in VARIANTS or settings.get("variant_formula") != VARIANTS[variant]:
        raise ValueError("Не совпадает режим предсказания в настройках калибровки")

    temperature = settings.get("temperature")

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("В настройках указана некорректная температура")

    if settings.get("weights_sha256") != weights_sha256:
        raise ValueError("Калибровка рассчитана для других весов; проверьте --weights")

    return settings


def _sha256(path: Path) -> str:
    """Зафиксировать содержимое входных файлов и готового сабмишена"""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Получить сабмишен с исходными или выбранными дообученными весами"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--read-workers", type=int, default=8)
    parser.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.read_workers < 1:
        parser.error("Размер пачки и число потоков должны быть положительными")

    if args.output.suffix.lower() != ".csv":
        parser.error("--output должен указывать на файл CSV")

    if args.calibration is not None and args.weights is None:
        parser.error("Вместе с --calibration укажите --weights")

    report_path = args.output.with_suffix(".json")

    for path in (args.output, report_path):
        if path.exists() and not args.overwrite:
            parser.error(f"Файл уже существует: {path}; для замены добавьте --overwrite")

    sample_path = args.test_dir / "sample_submission.csv"
    protected = [sample_path, args.weights, args.calibration]

    for path in (args.output, report_path):
        if any(item is not None and path.resolve() == item.resolve() for item in protected):
            parser.error("Пути результатов не должны совпадать с входными файлами")

    sample = read_sample(sample_path)
    image_dir = args.test_dir / "test" / "images"
    paths = [image_dir / f"{image_id}.png" for image_id in sample["image_id"]]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Не найден тестовый кроп: {path}")

    weights = args.weights if args.weights is not None else download_pretrained()
    weights_hash = _sha256(weights)
    calibration = None
    variant = args.variant or "single"
    temperature = 1.0

    if args.calibration is not None:
        calibration = load_calibration(args.calibration, weights_hash)

        if args.variant is not None and args.variant != calibration["variant"]:
            parser.error("--variant расходится с режимом сохранённой калибровки")

        variant = calibration["variant"]
        temperature = float(calibration["temperature"])

    if variant == "rotation" and args.batch_size < 2:
        parser.error("Для rotation нужен --batch-size >= 2")

    model = load_model(weights, device=args.device)
    device = paddle.device.get_device()
    cv2.setNumThreads(1)

    print(f"Устройство: {device}; веса: {weights}", flush=True)
    print(f"Режим: {variant}; температура: {temperature:.8f}", flush=True)
    print(
        f"Размер пачки: {args.batch_size}; потоки чтения: {args.read_workers}; "
        "FP32, RGB, вход 160 x 80",
        flush=True,
    )

    # Прогрев выполняется с тем же числом изображений, что и полная пачка.
    warmup_size = args.batch_size if variant == "single" else 2 * (args.batch_size // 2)
    warmup = np.repeat(_prepare_image(paths[0])[None], warmup_size, axis=0)

    for _ in range(3):
        predict_batch(model, warmup)

    del warmup

    if device.startswith("gpu"):
        paddle.device.synchronize()

    started = time.perf_counter()
    probabilities = predict_paths(
        model,
        paths,
        args.batch_size,
        args.read_workers,
        variant=variant,
        temperature=temperature,
    )
    submission = save_submission(sample, probabilities, args.output)

    if device.startswith("gpu"):
        paddle.device.synchronize()

    elapsed = time.perf_counter() - started
    report = {
        "weights": str(weights.resolve()),
        "weights_sha256": weights_hash,
        "variant": variant,
        "variant_formula": VARIANTS[variant],
        "temperature": temperature,
        "temperature_stage": "after_variant_probability",
        "preprocessing": PREPROCESSING,
        "calibration_path": str(args.calibration.resolve()) if args.calibration else None,
        "calibration": calibration,
        "sample": str(sample_path.resolve()),
        "sample_sha256": _sha256(sample_path),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "rows": len(submission),
        "device": device,
        "batch_size": args.batch_size,
        "read_workers": args.read_workers,
        "png_to_csv_seconds": round(elapsed, 3),
    }
    save_json(report_path, report)

    print(f"Строк: {len(submission)}; порядок ID: OK; вероятности: OK", flush=True)
    print(
        f"Диапазон p_180: {submission['p_180'].min():.10f} ... "
        f"{submission['p_180'].max():.10f}",
        flush=True,
    )
    print(f"PNG -> CSV: {elapsed:.2f} с; {len(paths) / elapsed:.1f} кропов/с", flush=True)
    print(f"CSV готов: {args.output}", flush=True)
    print(f"Настройки запуска: {report_path}", flush=True)


if __name__ == "__main__":
    main()
    