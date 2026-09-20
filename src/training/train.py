"""Дообучение ориентации текста с выбором весов по real-development Brier"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import paddle
from numpy.typing import NDArray
from paddle.nn import functional as F
from tqdm.auto import tqdm

from src.data.prepare import read_manifest
from src.data.preprocessing import MEAN, STD, read_rgb
from src.data.sources import save_json
from src.evaluation.evaluate import evaluate_model
from src.models.orientation import load_model

DEFAULTS = {
    "epochs": 6,
    "batch_size": 256,
    "lr": 0.001,
    "seed": 42,
    "selection_variant": "rotation",
}


def _sha256(path: Path) -> str:
    """Посчитать хеш файла для проверки данных и сохранённого состояния"""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_pair(path: Path) -> NDArray[np.uint8]:
    """Прочитать обе ориентации, выполняя поворот до изменения размера"""

    image = read_rgb(path)
    rotated = np.ascontiguousarray(image[::-1, ::-1])

    return np.stack([
        cv2.resize(view, (160, 80), interpolation=cv2.INTER_LINEAR)
        for view in (image, rotated)
    ])


def cache_training(paths: list[Path], workers: int) -> NDArray[np.uint8]:
    """Загрузить только train в RAM: две ориентации каждого исходного кропа"""

    images = np.empty((2 * len(paths), 80, 160, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in tqdm(range(0, len(paths), 256), desc="Train -> RAM"):
            chunk = paths[start : start + 256]

            for offset, pair in enumerate(pool.map(_read_pair, chunk)):
                position = 2 * (start + offset)
                images[position : position + 2] = pair

    return images


def prepare_batch(
    images: NDArray[np.uint8],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Нормализовать пачку; чётный индекс — класс 0, нечётный — класс 1"""

    pixels = images[indices].astype(np.float32) * np.float32(1 / 255)
    pixels = (pixels - MEAN) / STD
    values = np.ascontiguousarray(pixels.transpose(0, 3, 1, 2))
    labels = (indices % 2).astype(np.int64)

    return values, labels


def learning_rate(step: int, steps_per_epoch: int, settings: dict) -> float:
    """Разогрев четверть эпохи, затем cosine до 0.1 от начального LR"""

    total = settings["epochs"] * steps_per_epoch
    warmup = max(1, steps_per_epoch // 4)

    if not 0 <= step < total:
        raise ValueError("Шаг находится за пределами запланированного обучения")

    if step < warmup:
        return settings["lr"] * (step + 1) / warmup

    progress = (step - warmup) / max(1, total - warmup - 1)
    return settings["lr"] * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def train_epoch(model, optimizer, images, epoch: int, settings: dict) -> float:
    """Обучить одну эпоху, посетив каждую ориентацию ровно один раз"""

    epoch_seed = (settings["seed"] + epoch) % (2**31)
    paddle.seed(epoch_seed)
    rng = np.random.default_rng(epoch_seed)
    order = rng.permutation(len(images))
    steps = math.ceil(len(images) / settings["batch_size"])
    step = (epoch - 1) * steps
    total_loss = 0.0
    seen = 0
    model.train()
    optimizer.clear_grad()

    batches = tqdm(
        range(0, len(order), settings["batch_size"]),
        desc=f"Эпоха {epoch}/{settings['epochs']}",
        mininterval=2,
    )

    for start in batches:
        indices = order[start : start + settings["batch_size"]]
        values, labels = prepare_batch(images, indices)
        rate = learning_rate(step, steps, settings)
        optimizer.set_lr(rate)

        logits = model(paddle.to_tensor(values))
        loss = F.cross_entropy(logits, paddle.to_tensor(labels))
        value = float(loss.item())

        if not math.isfinite(value):
            raise RuntimeError("Train loss содержит NaN или бесконечность")

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        total_loss += value * len(indices)
        seen += len(indices)
        step += 1

        if step % 20 == 0:
            batches.set_postfix(loss=f"{total_loss / seen:.4f}", lr=f"{rate:.6f}")

    return total_loss / seen


def save_epoch(run: Path, model, optimizer, epoch: int, steps: int, history: list) -> None:
    """Опубликовать каталог эпохи после записи весов, оптимизатора и метаданных"""

    destination = run / f"epoch_{epoch:04d}"

    if destination.exists():
        raise FileExistsError(f"Эпоха уже сохранена: {destination}")

    with TemporaryDirectory(prefix=".checkpoint_", dir=run) as directory:
        temporary = Path(directory)
        paddle.save(model.state_dict(), str(temporary / "model.pdparams"))
        paddle.save(optimizer.state_dict(), str(temporary / "optimizer.pdopt"))
        save_json(temporary / "state.json", {
            "epoch": epoch,
            "global_step": epoch * steps,
            "config_sha256": _sha256(run / "config.json"),
            "weights_sha256": _sha256(temporary / "model.pdparams"),
            "optimizer_sha256": _sha256(temporary / "optimizer.pdopt"),
            "history": history,
        })
        temporary.rename(destination)


def latest_epoch(run: Path, steps: int, epochs: int) -> tuple[Path | None, list]:
    """Найти последнюю полностью записанную эпоху; незавершённая повторяется"""

    candidates = [
        path for path in run.glob("epoch_*")
        if path.is_dir() and path.name.removeprefix("epoch_").isdigit()
    ]

    if not candidates:
        return None, []

    latest = max(candidates, key=lambda path: int(path.name.removeprefix("epoch_")))
    state = json.loads((latest / "state.json").read_text(encoding="utf-8"))
    epoch = int(latest.name.removeprefix("epoch_"))

    if (
        state["epoch"] != epoch
        or not 1 <= epoch <= epochs
        or state["global_step"] != epoch * steps
        or state["config_sha256"] != _sha256(run / "config.json")
        or [row["epoch"] for row in state["history"]] != list(range(1, epoch + 1))
    ):
        raise ValueError("Сохранённая эпоха не соответствует настройкам запуска")

    for name, key in (
        ("model.pdparams", "weights_sha256"),
        ("optimizer.pdopt", "optimizer_sha256"),
    ):
        if _sha256(latest / name) != state[key]:
            raise ValueError(f"Повреждён файл состояния: {latest / name}")

    return latest, state["history"]


def publish_best(run: Path, history: list, variant: str) -> None:
    """Восстановить итоговые файлы из завершённых эпох и выбранного минимума Brier"""

    best = min(history, key=lambda row: (row["selection_brier"], row["epoch"]))
    source = run / f"epoch_{best['epoch']:04d}" / "model.pdparams"
    temporary = run / "best.pdparams.part"

    try:
        shutil.copyfile(source, temporary)
        temporary.replace(run / "best.pdparams")
    finally:
        temporary.unlink(missing_ok=True)

    save_json(run / "history.json", history)
    save_json(run / "best.json", {
        "epoch": best["epoch"],
        "criterion": "brier",
        "split": "development",
        "source": "real",
        "variant": variant,
        "temperature": 1.0,
        "brier": best["selection_brier"],
        "score": 1.0 - best["selection_brier"],
        "weights": "best.pdparams",
        "weights_sha256": _sha256(run / "best.pdparams"),
    })


def read_settings(args: argparse.Namespace) -> tuple[dict, Path]:
    """Создать настройки или восстановить их без изменения расписания LR"""

    if args.resume:
        if args.weights is not None:
            raise ValueError("При --resume веса и оптимизатор берутся из сохранённой эпохи")

        settings = json.loads((args.output / "config.json").read_text(encoding="utf-8"))

        for key in DEFAULTS:
            value = getattr(args, key)

            if value is not None and value != settings[key]:
                raise ValueError(f"При --resume нельзя менять {key}; создайте отдельный запуск")

        if settings["paddle"] != paddle.__version__:
            raise ValueError("Для --resume используйте то же окружение Paddle")

        manifest = args.manifest or Path(settings["manifest"])

        if _sha256(manifest) != settings["manifest_sha256"]:
            raise ValueError("Manifest изменился после начала обучения")

        return settings, manifest

    manifest = args.manifest or Path("data/external/manifest.csv")
    weights = args.weights or Path("weights/pretrained.pdparams")
    settings = {
        key: getattr(args, key) if getattr(args, key) is not None else value
        for key, value in DEFAULTS.items()
    }

    if (
        settings["epochs"] < 1
        or settings["batch_size"] < 2
        or not 0 < settings["lr"] < 1
        or not 0 <= settings["seed"] < 2**31
    ):
        raise ValueError("Нужны epochs >= 1, batch_size >= 2, 0 < lr < 1 и 0 <= seed < 2**31")

    settings.update({
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "initial_weights": str(weights.resolve()),
        "initial_weights_sha256": _sha256(weights),
        "loss": "cross_entropy",
        "optimizer": "Momentum, momentum=0.9, Nesterov, L2=4e-5, global_norm_clip=5",
        "schedule": "warmup 0.25 epoch, cosine to 0.1 * lr",
        "preprocessing": "RGB, rotation before resize, 160x80, ImageNet, FP32",
        "augmentation": "paired 0/180 degrees only",
        "epoch_seed": "(seed + epoch) % 2**31",
        "selection": "minimum Brier on real development; temperature=1",
        "resume": "last completed epoch; bitwise CUDA reproducibility is not guaranteed",
        "paddle": paddle.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    })

    return settings, manifest


def run_training(args: argparse.Namespace) -> None:
    """Обучить модель в отдельном каталоге или продолжить завершённую эпоху"""

    run = args.output
    settings, manifest = read_settings(args)
    frame = read_manifest(manifest)
    training = frame.loc[frame.split == "train"].reset_index(drop=True)
    development = frame.loc[frame.split == "development"].reset_index(drop=True)

    if training.empty or development.empty or not development.source.eq("real").any():
        raise ValueError("Нужны train и development с реальными кропами")

    paths = [manifest.parent / value for value in training.path]

    for value in [*training.path, *development.path]:
        if not (manifest.parent / value).is_file():
            raise FileNotFoundError(f"Не найден кроп: {manifest.parent / value}")

    if not args.resume:
        save_json(run / "config.json", settings)

    steps = math.ceil(2 * len(training) / settings["batch_size"])
    checkpoint, history = latest_epoch(run, steps, settings["epochs"])

    if history:
        publish_best(run, history, settings["selection_variant"])

        if len(history) == settings["epochs"]:
            print(f"Все эпохи уже завершены: {run / 'best.pdparams'}", flush=True)
            return

    weights = checkpoint / "model.pdparams" if checkpoint else Path(settings["initial_weights"])

    if checkpoint is None and _sha256(weights) != settings["initial_weights_sha256"]:
        raise ValueError("Изменились исходные веса")

    paddle.seed(settings["seed"])
    model = load_model(weights, device=args.device)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=settings["lr"],
        momentum=0.9,
        parameters=model.parameters(),
        weight_decay=4e-5,
        use_nesterov=True,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(5.0),
        name="orientation_momentum",
    )

    if checkpoint is not None:
        optimizer.set_state_dict(paddle.load(str(checkpoint / "optimizer.pdopt")))

    cv2.setNumThreads(1)
    print(f"Устройство: {paddle.device.get_device()}; каталог: {run}", flush=True)
    print(
        f"Train: {2 * len(training)}; development: {2 * len(development)}; "
        f"начальная эпоха: {len(history) + 1}/{settings['epochs']}",
        flush=True,
    )
    print(f"Кэш train: {2 * len(training) * 80 * 160 * 3 / 2**30:.2f} GiB", flush=True)
    print(f"Выбор: real-development Brier, {settings['selection_variant']}", flush=True)
    images = cache_training(paths, args.read_workers)
    initial_fc = model.fc.weight.numpy().copy()

    for epoch in range(len(history) + 1, settings["epochs"] + 1):
        started = time.perf_counter()
        loss = train_epoch(model, optimizer, images, epoch, settings)

        if epoch == 1 and np.array_equal(initial_fc, model.fc.weight.numpy()):
            raise RuntimeError("Параметры классификатора не изменились после эпохи")

        _, summary = evaluate_model(
            model, development, manifest.parent,
            batch_size=settings["batch_size"], read_workers=args.read_workers,
        )
        selected = summary.loc[
            (summary.source == "real") & (summary.variant == settings["selection_variant"])
        ]

        if len(selected) != 1:
            raise ValueError("Не найдена единственная метрика для выбора весов")

        record = {
            "epoch": epoch,
            "train_loss": loss,
            "selection_brier": float(selected.iloc[0].brier),
            "seconds": round(time.perf_counter() - started, 2),
            "development": summary.to_dict(orient="records"),
        }
        history.append(record)
        save_epoch(run, model, optimizer, epoch, steps, history)
        publish_best(run, history, settings["selection_variant"])
        print(
            f"Эпоха {epoch}: loss={loss:.6f}; "
            f"real-development Brier={record['selection_brier']:.8f}",
            flush=True,
        )
        print(summary.to_string(index=False, float_format="%.8f"), flush=True)

    best = json.loads((run / "best.json").read_text(encoding="utf-8"))
    print(f"Готово: {run / 'best.pdparams'}; лучшая эпоха: {best['epoch']}", flush=True)


def main() -> None:
    """Запустить обучение; --resume продолжает тот же план из --output"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/training"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--selection-variant", choices=("single", "rotation"))
    parser.add_argument("--read-workers", type=int, default=8)
    parser.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.read_workers < 1:
        parser.error("--read-workers должен быть положительным")

    if args.resume:
        if not (args.output / "config.json").is_file():
            parser.error("В --output нет настроек незавершённого запуска")
    elif args.output.exists():
        parser.error("Каталог уже существует; используйте --resume или другой --output")
    else:
        args.output.mkdir(parents=True, exist_ok=False)

    with (args.output / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("В этом каталоге уже работает обучение") from error

        # Имена параметров и состояний Momentum стабильны и при запуске из notebook.
        with paddle.utils.unique_name.guard("orientation_"):
            run_training(args)


if __name__ == "__main__":
    main()
    