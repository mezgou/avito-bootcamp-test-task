"""Подготовка реальных кропов RusTitW с проверкой направления через OCR"""

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from src.data.sources import (
    download,
    normalize_text,
    read_bgr,
    save_crop,
    save_json,
    script_group,
    unpack_single,
)
from src.data.split import image_signature, pixel_key

if TYPE_CHECKING:
    from src.data.recognition import TextRecognizer

KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "hardtype/rustitw-russian-language-visual-text-recognition/"
)
KAGGLE_SNAPSHOT = 2
MULTILINE = ("\n", "\r", r"\n", r"\/n", r"\r", "<br", "\u2028", "\u2029")


@dataclass
class Candidate:
    """Текст и прямоугольник кропа в пикселях исходной сцены"""

    crop_id: str
    text: str
    bounds: tuple[int, int, int, int]


@dataclass
class Scene:
    """Исходное изображение и выбранные аннотации"""

    scene_id: str
    path: Path
    url: str | None
    width: int
    height: int
    crops: list[Candidate]


def kaggle_url(name: str) -> str:
    """Построить адрес отдельного файла фиксированного снимка зеркала"""

    return KAGGLE_URL + quote(name, safe="") + (
        f"?datasetVersionNumber={KAGGLE_SNAPSHOT}"
    )


def read_scenes(
    cache: Path,
    limit: int,
    max_crops: int,
    seed: int,
    local_root: Path | None,
) -> tuple[list[Scene], dict]:
    """Прочитать реальные части корпуса и выбрать воспроизводимую подвыборку"""

    scenes = {}
    seen = {}
    counts = Counter()

    for part in ("train", "test"):
        relative = f"{part}/real/info.csv"

        if local_root is None:
            metadata = unpack_single(
                download(
                    kaggle_url(relative),
                    cache / f"rus_{part}.csv",
                    progress=True,
                    request_interval=1.0,
                )
            )
        else:
            metadata = local_root / relative

        with metadata.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                counts["scenes_in_metadata"] += 1
                name = row["image_name"]

                if not name or Path(name).name != name or "\\" in name:
                    raise ValueError(f"Некорректное имя изображения: {name!r}")

                width, height = int(row["width"]), int(row["height"])

                if min(width, height) <= 0:
                    raise ValueError(f"Некорректный размер сцены: {name}")

                annotation = json.loads(row["box_and_label"])
                signature = (width, height, annotation)

                if name in seen:
                    if seen[name] != signature:
                        raise ValueError(f"Конфликт аннотаций RusTitW: {name}")

                    counts["shared_scene_ids"] += 1
                    continue

                seen[name] = signature
                scene_id = "rus_" + Path(name).stem
                crops = []
                boxes = [box for group in annotation for box in group]

                for index, box in enumerate(boxes):
                    counts["boxes"] += 1
                    text = box["label"].strip()

                    if len(normalize_text(text)) < 3 or any(
                        marker in text.lower() for marker in MULTILINE
                    ):
                        counts["rejected_text"] += 1
                        continue

                    x, y = float(box["left"]), float(box["top"])
                    w, h = float(box["width"]), float(box["height"])

                    if (
                        not all(math.isfinite(v) for v in (x, y, w, h))
                        or min(x, y) < -1e-6
                        or x + w > 1.000001
                        or y + h > 1.000001
                        or w * width < 16
                        or h * height < 8
                        or not 1.5 <= w * width / (h * height) <= 40
                    ):
                        counts["rejected_geometry"] += 1
                        continue

                    bounds = (
                        max(0, math.floor(x * width)),
                        max(0, math.floor(y * height)),
                        min(width, math.ceil((x + w) * width)),
                        min(height, math.ceil((y + h) * height)),
                    )
                    crops.append(Candidate(f"{scene_id}_{index:04d}", text, bounds))

                if not crops:
                    continue

                if scene_id in scenes:
                    raise ValueError(f"Совпали ID разных изображений: {scene_id}")

                relative = f"{part}/real/images/{name}"
                path = (
                    local_root / relative
                    if local_root is not None
                    else cache / "images" / "rustitw" / name
                )
                scenes[scene_id] = Scene(
                    scene_id=scene_id,
                    path=path,
                    url=kaggle_url(relative) if local_root is None else None,
                    width=width,
                    height=height,
                    crops=crops,
                )

    selected = sorted(scenes.values(), key=lambda scene: scene.scene_id)
    counts["eligible_scenes"] = len(selected)
    random.Random(seed).shuffle(selected)
    selected = selected[:limit]

    for scene in selected:
        # Увеличение числа сцен не меняет выбор кропов в прежних сценах.
        if len(scene.crops) > max_crops:
            rng = random.Random(f"{seed}:{scene.scene_id}")
            scene.crops = rng.sample(scene.crops, max_crops)

        scene.crops.sort(key=lambda crop: crop.crop_id)

    return sorted(selected, key=lambda scene: scene.scene_id), dict(counts)


def annotation_key(scene: Scene) -> str:
    """Обнаружить изменение аннотаций перед повторным использованием кропов"""

    payload = {
        "scene_id": scene.scene_id,
        "width": scene.width,
        "height": scene.height,
        "crops": [asdict(crop) for crop in scene.crops],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()

    return hashlib.sha256(encoded).hexdigest()


def load_scene(scene: Scene) -> NDArray[np.uint8]:
    """Получить сцену из локального корпуса или кэша скачивания"""

    path = scene.path

    if scene.url is not None:
        path = unpack_single(
            download(scene.url, path, request_interval=1.0)
        )

    image = read_bgr(path)

    if image.shape[:2] != (scene.height, scene.width):
        raise ValueError(f"Размер сцены отличается от аннотации: {scene.scene_id}")

    return image


def process_scene(
    scene: Scene,
    image: NDArray[np.uint8],
    output: Path,
    recognizer: "TextRecognizer",
    batch_size: int,
) -> dict:
    """Принять только кропы с однозначным совпадением OCR и транскрипции"""

    state = {
        "annotation_key": annotation_key(scene),
        "scene": {
            "scene_id": scene.scene_id,
            "source": "real",
            "dataset": "rustitw",
            "signature": image_signature(image),
        },
        "crops": [],
        "rejected": [],
    }
    candidates = []
    ocr_images = []

    for candidate in scene.crops:
        left, top, right, bottom = candidate.bounds
        pixels = np.ascontiguousarray(image[top:bottom, left:right])
        candidates.append((candidate, pixels))
        ocr_images.extend((pixels, cv2.rotate(pixels, cv2.ROTATE_180)))

    texts = recognizer.recognize(ocr_images, batch_size)

    if len(texts) != len(ocr_images):
        raise ValueError("Число результатов OCR не совпало с числом изображений")

    for index, (candidate, pixels) in enumerate(candidates):
        first, second = texts[2 * index : 2 * index + 2]
        target = normalize_text(candidate.text)
        matches = (
            normalize_text(first) == target,
            normalize_text(second) == target,
        )

        if sum(matches) != 1:
            state["rejected"].append(
                {
                    "crop_id": candidate.crop_id,
                    "text": candidate.text,
                    "reason": "both_match" if all(matches) else "no_exact_match",
                    "ocr_0": first,
                    "ocr_180": second,
                }
            )
            continue

        rotation = 180 if matches[1] else 0

        if rotation:
            pixels = cv2.rotate(pixels, cv2.ROTATE_180)

        relative = f"crops/{candidate.crop_id}.png"
        save_crop(output / relative, pixels)
        state["crops"].append(
            {
                "crop_id": candidate.crop_id,
                "path": relative,
                "scene_id": scene.scene_id,
                "source": "real",
                "dataset": "rustitw",
                "kind": "line",
                "text": candidate.text,
                "script": script_group(candidate.text),
                "width": pixels.shape[1],
                "height": pixels.shape[0],
                "pixel_key": pixel_key(pixels),
                "orientation_method": "exact_transcription",
                "applied_rotation": rotation,
                "ocr_0": first,
                "ocr_180": second,
            }
        )

    return state


def prepare_rustitw(
    output: Path,
    cache: Path,
    limit: int = 1000,
    max_crops: int = 20,
    seed: int = 42,
    local_root: Path | None = None,
    device: str = "cpu",
    batch_size: int = 16,
    workers: int = 2,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Подготовить RusTitW и вернуть кропы, сцены и статистику без split"""

    if min(limit, max_crops, batch_size, workers) < 1 or seed < 0:
        raise ValueError("Лимиты должны быть положительными, seed — неотрицательным")

    if device != "cpu" and re.fullmatch(r"gpu:\d+", device) is None:
        raise ValueError("Устройство OCR: cpu или gpu:0")

    # Paddle нужен только при явном выборе RusTitW в общей подготовке.
    from src.data.recognition import MODEL_REVISION, TextRecognizer

    settings = {
        "snapshot": KAGGLE_SNAPSHOT,
        "max_crops": max_crops,
        "seed": seed,
        "recognizer_commit": MODEL_REVISION,
        "device": device,
        "batch_size": batch_size,
    }
    metadata = output / "preparation_cache" / "rustitw"
    metadata.mkdir(parents=True, exist_ok=True)
    settings_path = metadata / "settings.json"

    if settings_path.is_file():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))

        if previous != settings:
            raise ValueError("Изменились настройки RusTitW; нужен --rebuild")
    else:
        if any(metadata.glob("*.json")):
            raise ValueError("Записи RusTitW без настроек; нужен --rebuild")

        save_json(settings_path, settings)

    scenes, counts = read_scenes(cache, limit, max_crops, seed, local_root)

    if not scenes:
        raise ValueError("В RusTitW нет подходящих сцен")

    records = {}
    pending = []

    for scene in scenes:
        saved = metadata / f"{scene.scene_id}.json"

        if saved.is_file():
            state = json.loads(saved.read_text(encoding="utf-8"))

            if state.get("annotation_key") != annotation_key(scene):
                raise ValueError(f"Изменилась аннотация {scene.scene_id}; нужен --rebuild")

            if all(
                (output / row["path"]).is_file()
                and (output / row["path"]).stat().st_size > 0
                for row in state["crops"]
            ):
                records[scene.scene_id] = state
                continue

        pending.append(scene)

    print(
        f"RusTitW: выбрано сцен {len(scenes)}, осталось обработать {len(pending)}",
        flush=True,
    )
    recognizer = TextRecognizer(cache / "cyrillic_rec", device) if pending else None

    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=len(pending), desc="Кропы RusTitW", unit="scene") as bar,
    ):
        for start in range(0, len(pending), workers):
            chunk = pending[start : start + workers]

            # OCR одной сцены не зависит от соседних сцен в очереди.
            for scene, image in zip(chunk, pool.map(load_scene, chunk), strict=True):
                state = process_scene(scene, image, output, recognizer, batch_size)
                save_json(metadata / f"{scene.scene_id}.json", state)
                records[scene.scene_id] = state
                bar.update(1)

    ordered = [records[scene.scene_id] for scene in scenes]
    rows = [row for record in ordered for row in record["crops"]]

    if not rows:
        raise ValueError("OCR не принял ни одного кропа RusTitW")

    rejected = Counter(
        item["reason"] for record in ordered for item in record["rejected"]
    )
    stats = {
        "dataset": "rustitw",
        "settings": settings,
        "source_url": KAGGLE_URL,
        "local_root": str(local_root.resolve()) if local_root else None,
        "metadata_counts": counts,
        "requested_scenes": limit,
        "selected_scenes": len(scenes),
        "scenes_with_crops": sum(bool(record["crops"]) for record in ordered),
        "base_crops": len(rows),
        "ocr_rejected": dict(rejected),
        "rotated_to_upright": sum(row["applied_rotation"] == 180 for row in rows),
        "selection": (
            "Точное совпадение нормализованной транскрипции "
            "только в одном направлении"
        ),
    }
    frame = pd.DataFrame(rows).sort_values("crop_id").reset_index(drop=True)

    return frame, [record["scene"] for record in ordered], stats
