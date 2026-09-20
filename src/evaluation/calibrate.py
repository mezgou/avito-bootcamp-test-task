"""Подбор температуры по Brier на реальной части calibration"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit

from src.evaluation.metrics import binary_metrics, summarize_metrics

TEMPERATURE_BOUNDS = (0.25, 4.0)
GRID_POINTS = 81
VARIANTS = {
    "single": "p(x)",
    "rotation": "(p(x) + 1 - p(rot180(x))) / 2",
}


def apply_temperature(
    probabilities: ArrayLike,
    temperature: float,
) -> NDArray[np.float64]:
    """Преобразовать вероятности: sigmoid(logit(p) / temperature)

    Для rotation сначала усредняются вероятности двух проходов, затем
    применяется температура. При T=1 значения сохраняются точно.
    Граничные p=0 и p=1 остаются неизменными: исходные logits неизвестны.
    """

    values = np.asarray(probabilities)

    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "biuf":
        raise ValueError("Нужен непустой одномерный массив вероятностей")

    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Вероятности должны быть конечными числами от 0 до 1")

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Температура должна быть положительным конечным числом")

    result = values.astype(np.float64, copy=True)

    if temperature != 1.0:
        interior = (result > 0) & (result < 1)
        values = result[interior]
        logits = np.log(values) - np.log1p(-values)
        result[interior] = expit(logits / temperature)

    return result


def fit_temperature(
    targets: ArrayLike,
    probabilities: ArrayLike,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> dict:
    """Подобрать T на сетке log(T) и уточнить лучший участок по Brier

    Границы и T=1 всегда входят в кандидаты. Если улучшение относительно
    T=1 не превышает 1e-12, преобразование отключается. Это сравнение
    на данных подбора, а не оценка качества на независимой выборке.
    """

    before = binary_metrics(targets, probabilities)
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    lower, upper = bounds

    if not np.isfinite(bounds).all() or not 0 < lower <= 1 <= upper or lower >= upper:
        raise ValueError("Нужны конечные границы 0 < min <= 1 <= max и min < max")

    if np.unique(targets).size != 2:
        raise ValueError("Для калибровки нужны оба класса ориентации")

    def objective(log_temperature: float) -> float:
        calibrated = apply_temperature(probabilities, float(np.exp(log_temperature)))
        return float(np.mean((calibrated - targets) ** 2))

    grid = np.unique(np.r_[np.linspace(np.log(lower), np.log(upper), GRID_POINTS), 0.0])
    losses = np.array([objective(value) for value in grid])
    best = int(np.argmin(losses))
    left = grid[max(0, best - 1)]
    right = grid[min(len(grid) - 1, best + 1)]

    # Brier не обязан быть унимодальным по T, поэтому сначала нужна сетка.
    refined = minimize_scalar(
        objective,
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )

    if not refined.success or not np.isfinite(refined.fun):
        raise RuntimeError(f"Не удалось уточнить температуру: {refined.message}")

    candidates = (0.0, float(grid[best]), float(refined.x))
    chosen = min(candidates, key=lambda value: (objective(value), abs(value)))
    temperature = float(np.clip(np.exp(chosen), lower, upper))
    after = binary_metrics(targets, apply_temperature(probabilities, temperature))

    if before["brier"] - after["brier"] <= 1e-12:
        temperature = 1.0
        after = before.copy()

    return {
        "temperature": temperature,
        "bounds": [float(lower), float(upper)],
        "grid_points": len(grid),
        "at_bound": bool(np.isclose(temperature, bounds, rtol=0, atol=1e-6).any()),
        "before": before,
        "after": after,
        "brier_improvement": before["brier"] - after["brier"],
    }


def read_calibration(
    directory: Path,
    variant: str,
) -> tuple[pd.DataFrame, dict]:
    """Прочитать результаты evaluate и выбрать real для заданного режима"""

    if variant not in VARIANTS:
        raise ValueError("Неизвестный режим предсказания")

    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))

    if report.get("split") != "calibration" or report.get("temperature") != 1.0:
        raise ValueError("Нужен результат evaluate на calibration без калибровки")

    if report.get("variants", {}).get(variant) != VARIANTS[variant]:
        raise ValueError("Формула режима не совпадает с ожидаемой")

    if report.get("labels") != {"0": "upright", "1": "rotated_180"}:
        raise ValueError("Не совпадает соответствие классов ориентациям")

    for key in ("weights_sha256", "manifest_sha256"):
        value = report.get(key, "")

        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError(f"В отчёте отсутствует корректный {key}")

    predictions = pd.read_csv(
        directory / "predictions.csv.gz",
        keep_default_na=False,
        float_precision="round_trip",
    )
    summary = summarize_metrics(predictions)

    if "variant" not in predictions or not predictions.split.eq("calibration").all():
        raise ValueError("Ожидались прогнозы режимов только для calibration")

    # Исключаем случайное объединение CSV и отчёта от разных запусков.
    keys = ["split", "source", "variant"]
    columns = ["crops", "n", "brier", "score", "log_loss", "accuracy"]
    actual = summary.set_index(keys).sort_index()[columns]
    recorded = pd.DataFrame(report["metrics"]).set_index(keys).sort_index()[columns]

    if not actual.index.equals(recorded.index) or not np.allclose(
        actual.to_numpy(dtype=float), recorded.to_numpy(dtype=float),
        rtol=1e-10, atol=1e-12,
    ):
        raise ValueError("Метрики CSV не совпадают с report.json")

    selected = predictions.loc[
        (predictions.source == "real") & (predictions.variant == variant)
    ].copy().reset_index(drop=True)

    if selected.empty:
        raise ValueError("В calibration нет реальных кропов для выбранного режима")

    pairs = selected.groupby("crop_id").target.agg(["size", "nunique"])

    if not pairs.eq(2).all().all():
        raise ValueError("Для каждого кропа нужны обе ориентации ровно по одному разу")

    if variant == "rotation":
        sums = selected.groupby("crop_id").p_180.sum().to_numpy()

        if not np.allclose(sums, 1.0, rtol=0, atol=1e-12):
            raise ValueError("Вероятности rotation для пары должны давать сумму 1")

    return selected, report


def _sha256(path: Path) -> str:
    """Сохранить связь настройки с конкретными входными файлами"""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Подобрать температуру выбранного режима и сохранить настройку в JSON"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--temperature-bounds", nargs=2, type=float,
        default=TEMPERATURE_BOUNDS, metavar=("MIN", "MAX"),
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Файл уже существует; укажите новый --output")

    frame, evaluation = read_calibration(args.evaluation, args.variant)
    fitted = fit_temperature(
        frame.target.to_numpy(),
        frame.p_180.to_numpy(),
        bounds=tuple(args.temperature_bounds),
    )
    result = {
        "method": "temperature",
        "variant": args.variant,
        "variant_formula": VARIANTS[args.variant],
        "apply_to": "variant_probability",
        "formula": "sigmoid(logit(p) / temperature)",
        "fit_split": "calibration",
        "fit_source": "real",
        "criterion": "brier",
        "weights": evaluation["weights"],
        "weights_sha256": evaluation["weights_sha256"],
        "manifest_sha256": evaluation["manifest_sha256"],
        "preprocessing": evaluation["preprocessing"],
        "crops": int(frame.crop_id.nunique()),
        "evaluation": str(args.evaluation.resolve()),
        "input_sha256": {
            name: _sha256(args.evaluation / name)
            for name in ("predictions.csv.gz", "report.json")
        },
        **fitted,
    }

    from src.data.sources import save_json

    save_json(args.output, result)
    print(
        f"calibration / real / {args.variant}: "
        f"кропов {result['crops']}; примеров {fitted['before']['n']}",
        flush=True,
    )
    print(f"Температура: {fitted['temperature']:.8f}", flush=True)
    print(
        pd.DataFrame([
            {"stage": "before", **fitted["before"]},
            {"stage": "after", **fitted["after"]},
        ]).to_string(index=False, float_format="%.8f"),
        flush=True,
    )

    if fitted["at_bound"]:
        print("Выбранная температура находится на границе заданного диапазона", flush=True)

    print("Метрики before/after посчитаны на данных подбора температуры", flush=True)
    print(f"Настройка сохранена: {args.output}", flush=True)


if __name__ == "__main__":
    main()
    