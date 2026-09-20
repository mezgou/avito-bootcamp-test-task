"""Единая подготовка HierText, синтетики и необязательного RusTitW"""

import argparse
import fcntl
import json
import os
import shutil
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.sources import read_bgr, save_json
from src.data.split import SPLITS, assign_splits, pixel_key

DATASETS = ("hiertext", "synthetic", "rustitw")
RESULT_FILES = (
    "manifest.csv",
    "split_groups.csv",
    "split_summary.csv",
    "preparation_report.json",
    "preparation_settings.json",
)


@contextmanager
def output_lock(output: Path):
    """Исключить одновременное изменение одного корпуса двумя процессами"""

    output.mkdir(parents=True, exist_ok=True)

    with (output / ".prepare.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Подготовка уже работает в {output}") from error

        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def rebuild_output(output: Path, protected: list[Path]) -> None:
    """Очистить результаты подготовки, сохранив исходные данные и шрифты"""

    target = output.resolve()

    if target in (Path("/"), Path.home().resolve(), Path.cwd().resolve()):
        raise ValueError("--output должен указывать на отдельный каталог данных")

    for path in protected:
        if path.resolve().is_relative_to(target):
            raise ValueError(f"Перед --rebuild вынесите исходные данные из {output}: {path}")

    for name in ("crops", "preparation_cache"):
        path = output / name

        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    for name in RESULT_FILES:
        (output / name).unlink(missing_ok=True)
        (output / (name + ".part")).unlink(missing_ok=True)

    print(f"Результаты в {output} очищены; кэш и шрифты сохранены", flush=True)


def read_manifest(path: Path) -> pd.DataFrame:
    """Прочитать manifest и проверить формат, пути и отсутствие утечек"""

    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "crop_id", "path", "scene_id", "group_id", "split", "source", "dataset",
        "kind", "text", "script", "width", "height", "pixel_key",
        "orientation_method", "applied_rotation",
    }

    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Manifest пуст или имеет другой формат; нужен --rebuild")

    for column in ("crop_id", "path", "pixel_key"):
        if frame[column].duplicated().any():
            raise ValueError(f"Дубликаты в столбце {column}")

    for column in ("crop_id", "scene_id", "group_id", "pixel_key"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"Некорректные значения в {column}")

    if not frame.split.isin(SPLITS).all():
        raise ValueError("Неизвестная часть данных в manifest")

    if not frame.dataset.isin(DATASETS).all():
        raise ValueError("Неизвестный датасет в manifest")

    if not frame.source.isin(("real", "synthetic")).all() or not (
        (frame.source == "synthetic") == (frame.dataset == "synthetic")
    ).all():
        raise ValueError("Тип источника расходится с датасетом")

    sizes = frame[["width", "height"]].to_numpy(dtype=float)

    if (
        not np.isfinite(sizes).all()
        or (sizes <= 0).any()
        or (sizes != np.floor(sizes)).any()
    ):
        raise ValueError("Размеры кропов должны быть положительными целыми числами")

    if not frame.applied_rotation.isin((0, 180)).all():
        raise ValueError("Неизвестный поворот исходного кропа")

    for column in ("scene_id", "group_id"):
        if frame.groupby(column).split.nunique().max() != 1:
            raise ValueError(f"Пересечение частей по {column}")

    for primary, alias in (
        ("scene_id", "scene"), ("group_id", "group"), ("script", "language"),
    ):
        if alias in frame and not frame[primary].equals(frame[alias]):
            raise ValueError(f"Расходятся столбцы {primary} и {alias}")

    for row in frame.itertuples(index=False):
        relative = Path(row.path)

        if (
            relative.parts != ("crops", f"{row.crop_id}.png")
            or "\\" in row.path
            or not (path.parent / relative).resolve().is_relative_to(path.parent.resolve())
        ):
            raise ValueError(f"Ожидался относительный путь crops/ID.png: {row.path}")

    return frame


def check_files(frame: pd.DataFrame, output: Path, sample_size: int) -> None:
    """Проверить наличие всех PNG, размеры и хеши выбранных изображений"""

    missing = [
        value for value in frame.path
        if not (output / value).is_file() or (output / value).stat().st_size == 0
    ]

    if missing:
        raise FileNotFoundError(f"Отсутствуют {len(missing)} кропов: {missing[:5]}")

    sample = frame.sample(n=min(sample_size, len(frame)), random_state=42)

    for row in sample.itertuples(index=False):
        image = read_bgr(output / row.path)

        if image.shape[:2] != (row.height, row.width) or pixel_key(image) != row.pixel_key:
            raise ValueError(f"Изображение отличается от manifest: {row.path}")


def check_data(output: Path, sample_size: int) -> None:
    """Проверить готовый корпус без загрузки источников и распознавателя"""

    frame = read_manifest(output / "manifest.csv")
    check_files(frame, output, sample_size)
    groups = pd.read_csv(output / "split_groups.csv", keep_default_na=False)

    if groups.scene_id.duplicated().any():
        raise ValueError("В split_groups.csv повторяются сцены")

    if groups.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("В split_groups.csv группа попала в разные части")

    lookup = groups.set_index("scene_id")

    for column in ("source", "dataset", "group_id", "split"):
        if not frame[column].eq(frame.scene_id.map(lookup[column])).all():
            raise ValueError(f"Manifest и split_groups.csv расходятся по {column}")

    absent = set(SPLITS) - set(frame.split)

    if absent:
        raise ValueError(f"Для обучения и оценки не хватает частей: {sorted(absent)}")

    print(frame.groupby(["split", "dataset", "script"]).size().to_string(), flush=True)
    print(
        f"Проверка пройдена: {len(frame)} кропов; все файлы существуют; "
        f"декодировано PNG: {min(sample_size, len(frame))}",
        flush=True,
    )


def build_settings(args: argparse.Namespace, previous: dict) -> dict:
    """Сохранить выбранные источники и настройки при повторном запуске"""

    selected = args.sources or previous.get("sources", ["hiertext", "synthetic"])
    settings = {
        "sources": [name for name in DATASETS if name in selected],
        "seed": args.seed if args.seed is not None else previous.get("seed", 42),
    }

    def option(source: str, key: str, argument: str, default):
        value = getattr(args, argument)
        return value if value is not None else previous.get(source, {}).get(key, default)

    if "hiertext" in selected:
        settings["hiertext"] = {
            "max_crops": option("hiertext", "max_crops", "max_crops", 48),
        }

    if "synthetic" in selected:
        fonts_dir = option("synthetic", "fonts_dir", "fonts_dir", "data/fonts")
        settings["synthetic"] = {
            "count": option("synthetic", "count", "synthetic_count", 4000),
            "fonts_dir": str(Path(fonts_dir).resolve()),
        }

    if "rustitw" in selected:
        root = option("rustitw", "local_root", "rustitw_root", None)
        settings["rustitw"] = {
            "limit": option("rustitw", "limit", "rus_scenes", 1000),
            "max_crops": option("rustitw", "max_crops", "rus_max_crops", 20),
            "device": option("rustitw", "device", "ocr_device", os.getenv("DEVICE", "cpu")),
            "batch_size": option("rustitw", "batch_size", "ocr_batch_size", 16),
            "local_root": str(Path(root).resolve()) if root else None,
        }

    if settings["seed"] < 0:
        raise ValueError("Seed должен быть неотрицательным")

    for name in settings["sources"]:
        for key in ("max_crops", "count", "limit", "batch_size"):
            if key in settings[name] and settings[name][key] < 1:
                raise ValueError(f"{name}.{key} должен быть положительным")

    if previous and not args.rebuild:
        if (
            settings["seed"] != previous["seed"]
            or not set(previous["sources"]).issubset(selected)
        ):
            raise ValueError("Изменение seed или удаление источника требует --rebuild")

        for name in previous["sources"]:
            for key, value in previous[name].items():
                if key in ("fonts_dir", "local_root"):
                    continue

                actual = settings[name][key]

                if (
                    (key == "limit" and actual < value)
                    or (key != "limit" and actual != value)
                ):
                    raise ValueError(f"Изменение {name}.{key} требует --rebuild")

    return settings


def prepare_sources(
    output: Path,
    cache: Path,
    settings: dict,
    workers: int,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Подключить только выбранные модули и собрать общий состав корпуса"""

    frames = []
    scenes = []
    stats = {}

    for name in settings["sources"]:
        options = settings[name].copy()
        print(f"Подготовка источника: {name}", flush=True)

        if name == "hiertext":
            from src.data.hiertext import prepare_hiertext

            result = prepare_hiertext(output, cache, seed=settings["seed"], **options)

        elif name == "synthetic":
            from src.data.synthetic import prepare_synthetic

            options["fonts_dir"] = Path(options["fonts_dir"])
            result = prepare_synthetic(output, seed=settings["seed"], **options)

        else:
            from src.data.rustitw import prepare_rustitw

            if options["local_root"] is not None:
                options["local_root"] = Path(options["local_root"])

            result = prepare_rustitw(
                output, cache, seed=settings["seed"], workers=workers, **options
            )

        frame, source_scenes, source_stats = result
        frames.append(frame)
        scenes.extend(source_scenes)
        stats[name] = source_stats

    return pd.concat(frames, ignore_index=True).fillna(""), scenes, stats


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Сохранить CSV через временный файл"""

    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_results(
    frame: pd.DataFrame,
    groups: pd.DataFrame,
    output: Path,
    settings: dict,
    stats: dict,
    elapsed: float,
) -> None:
    """Сохранить статистику, отчёт и затем готовый manifest"""

    frame = frame.assign(
        scene=frame.scene_id,
        group=frame.group_id,
        language=frame.script,
        small=frame.height <= 32,
        long=frame.width / frame.height > 10,
    )
    summary = (
        frame.groupby(["split", "source", "dataset", "script"], sort=True)
        .agg(
            crops=("crop_id", "size"),
            scenes=("scene_id", "nunique"),
            groups=("group_id", "nunique"),
            small=("small", "sum"),
            long=("long", "sum"),
        )
        .reset_index()
    )
    summary["orientation_examples"] = summary.crops * 2
    dependencies = {}

    for package in (
        "numpy", "pandas", "opencv-python", "opencv-python-headless",
        "Pillow", "paddlepaddle", "paddlepaddle-gpu",
    ):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            continue

    real_cyrillic = frame[
        (frame.source == "real") & frame.script.isin(("cyrillic", "mixed"))
    ]
    report = {
        "settings": settings,
        "base_crops": len(frame),
        "crops_by_dataset": frame.groupby("dataset").size().to_dict(),
        "crops_by_split": frame.groupby("split").size().to_dict(),
        "real_cyrillic_by_split": real_cyrillic.groupby("split").size().to_dict(),
        "train_orientation_examples": 2 * int((frame.split == "train").sum()),
        "path_base": "Directory containing manifest.csv",
        "orientation": "PNG хранят нормальный текст; пары 0/180 создаются при обучении",
        "applied_rotation": "Исправление исходного кропа, не целевая метка модели",
        "split": "70/10/10/10 по группам; при расширении прежние назначения сохранены",
        "preparation": stats,
        "elapsed_seconds": round(elapsed, 2),
        "dependencies": dependencies,
        "limitations": [
            "Синтетика не заменяет оценку на реальном русском тексте",
            "OCR-фильтр RusTitW предпочитает читаемые надписи",
            "После изучения holdout пересборка не делает его независимой оценкой",
        ],
    }
    temporary = output / "manifest.csv.part"
    frame.to_csv(temporary, index=False)
    verified = read_manifest(temporary)
    check_files(verified, output, sample_size=0)

    save_csv(groups, output / "split_groups.csv")
    save_csv(summary, output / "split_summary.csv")
    save_json(output / "preparation_report.json", report)

    # При ошибке подготовки прежний manifest остаётся доступен.
    temporary.replace(output / "manifest.csv")
    print(summary.to_string(index=False), flush=True)
    print(f"Готово: {output / 'manifest.csv'}", flush=True)


def main() -> None:
    """Подготовить выбранные источники, продолжить работу или проверить результат"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--sources", nargs="+", choices=DATASETS,
        help="Источники корпуса; впервые: hiertext synthetic, повторно: сохранённый выбор",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rebuild", action="store_true",
        help="Пересоздать кропы и split с настройками команды, сохранив кэш и шрифты",
    )
    mode.add_argument(
        "--check", action="store_true",
        help="Проверить готовые данные без подготовки",
    )
    parser.add_argument("--check-images", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--max-crops", type=int, help="Кропы HierText на сцену; по умолчанию 48",
    )
    parser.add_argument(
        "--synthetic-count", type=int,
        help="Число синтетических кропов; по умолчанию 4000",
    )
    parser.add_argument(
        "--fonts-dir", type=Path, help="Каталог шрифтов; по умолчанию data/fonts",
    )
    parser.add_argument("--rus-scenes", type=int, help="Сцены RusTitW; по умолчанию 1000")
    parser.add_argument(
        "--rus-max-crops", type=int, help="Кропы RusTitW на сцену; по умолчанию 20",
    )
    parser.add_argument(
        "--rustitw-root", type=Path, help="Локальный корпус с train/real и test/real",
    )
    parser.add_argument(
        "--ocr-device", help="cpu или gpu:0; по умолчанию DEVICE из окружения или cpu",
    )
    parser.add_argument(
        "--ocr-batch-size", type=int, help="Размер пачки OCR; по умолчанию 16",
    )
    parser.add_argument("--workers", type=int, default=2, help="Потоки загрузки сцен RusTitW")
    args = parser.parse_args()

    if args.check_images < 0 or args.workers < 1:
        parser.error("Число проверяемых PNG неотрицательно, число потоков положительно")

    cv2.setNumThreads(1)

    with output_lock(args.output):
        if args.check:
            check_data(args.output, args.check_images)
            return

        settings_path = args.output / "preparation_settings.json"
        previous_settings = {}

        if settings_path.is_file() and not args.rebuild:
            previous_settings = json.loads(settings_path.read_text(encoding="utf-8"))

            if not isinstance(previous_settings, dict) or "sources" not in previous_settings:
                raise ValueError("Некорректные настройки подготовки; нужен --rebuild")

        settings = build_settings(args, previous_settings)

        if args.rebuild:
            protected = [args.cache, args.fonts_dir or Path("data/fonts")]

            if "synthetic" in settings:
                protected.append(Path(settings["synthetic"]["fonts_dir"]))

            if "rustitw" in settings and settings["rustitw"]["local_root"]:
                protected.append(Path(settings["rustitw"]["local_root"]))

            rebuild_output(args.output, protected)

        elif not previous_settings and any(
            (args.output / name).exists()
            for name in ("manifest.csv", "crops", "preparation_cache")
        ):
            raise ValueError("В каталоге есть результаты без настроек; нужен --rebuild")

        manifest = args.output / "manifest.csv"
        previous = read_manifest(manifest) if manifest.is_file() else None
        save_json(settings_path, settings)
        started = time.perf_counter()
        crops, scenes, stats = prepare_sources(
            args.output, args.cache, settings, args.workers
        )

        print("Объединение дубликатов и разделение сцен", flush=True)
        frame, groups, split_stats = assign_splits(
            crops, scenes, seed=settings["seed"], previous=previous
        )
        stats["split"] = split_stats
        save_results(
            frame, groups, args.output, settings, stats,
            elapsed=time.perf_counter() - started,
        )


if __name__ == "__main__":
    main()
    