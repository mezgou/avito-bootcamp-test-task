"""Подготовка слов и строк из архива HierText validation"""

import gzip
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from src.data.sources import (
    download,
    read_bgr,
    save_crop,
    save_json,
    script_group,
)
from src.data.split import image_signature, pixel_key

HIER_REVISION = "70b6620b2b112597d8219e11eee9773a1403827c"
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    f"hiertext/{HIER_REVISION}/gt/validation.jsonl.gz"
)
ARCHIVE_URL = "https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz"


def eligible_word(word: dict) -> bool:
    """Проверить читаемость и пригодность слова"""

    text = word.get("text", "").strip()

    return bool(
        word.get("legible", False)
        and not word.get("vertical", False)
        and not word.get("handwritten", False)
        and len(word.get("vertices", [])) == 4
        and len(text) >= 2
        and re.search(r"[A-Za-zА-Яа-яЁё0-9]", text)
    )


def line_quad(line: dict) -> NDArray[np.float32] | None:
    """Построить строку по согласованным направлениям её слов"""

    words = line.get("words", [])

    if len(words) < 2 or not all(eligible_word(word) for word in words):
        return None

    try:
        quads = np.asarray([word["vertices"] for word in words], np.float32)
    except (TypeError, ValueError):
        return None

    if quads.shape != (len(words), 4, 2) or not np.isfinite(quads).all():
        return None

    axes = (quads[:, 1] - quads[:, 0] + quads[:, 2] - quads[:, 3]) / 2
    norms = np.linalg.norm(axes, axis=1)

    if not np.isfinite(norms).all() or (norms < 2).any():
        return None

    axes = axes / norms[:, None]
    horizontal = axes.mean(0)
    length = np.linalg.norm(horizontal)

    if length < 1e-6:
        return None

    horizontal = horizontal / length

    if (axes @ horizontal < 0.97).any():
        return None

    vertical = np.array([-horizontal[1], horizontal[0]])

    if ((quads[:, 3] - quads[:, 0]) @ vertical <= 0).any():
        return None

    centers = quads.mean(1)
    heights = np.linalg.norm(quads[:, 3] - quads[:, 0], axis=1)

    if (np.diff(centers @ horizontal) <= 0).any():
        return None

    if np.ptp(centers @ vertical) > 0.75 * np.median(heights):
        return None

    points = quads.reshape(-1, 2)
    xx, yy = points @ horizontal, points @ vertical

    return np.asarray(
        [
            x * horizontal + y * vertical
            for x, y in (
                (xx.min(), yy.min()),
                (xx.max(), yy.min()),
                (xx.max(), yy.max()),
                (xx.min(), yy.max()),
            )
        ],
        dtype=np.float32,
    )


def upright_crop(
    image: NDArray[np.uint8],
    vertices: list | NDArray[np.float32],
) -> NDArray[np.uint8] | None:
    """Выпрямить кроп по вершинам, упорядоченным относительно текста"""

    try:
        quad = np.asarray(vertices, np.float32)
    except (TypeError, ValueError):
        return None

    if (
        quad.shape != (4, 2)
        or not np.isfinite(quad).all()
        or not cv2.isContourConvex(quad)
    ):
        return None

    height, width = image.shape[:2]

    if (
        (quad[:, 0] < 0).any()
        or (quad[:, 0] > width - 1).any()
        or (quad[:, 1] < 0).any()
        or (quad[:, 1] > height - 1).any()
    ):
        return None

    u = (quad[1] - quad[0] + quad[2] - quad[3]) / 2
    v = (quad[3] - quad[0] + quad[2] - quad[1]) / 2

    # Положительный определитель исключает зеркальный порядок вершин
    if u[0] * v[1] - u[1] * v[0] <= 0:
        return None

    width = round(
            (
                np.linalg.norm(quad[1] - quad[0])
                + np.linalg.norm(quad[2] - quad[3])
            )
            / 2
        )
    height = round(
            (
                np.linalg.norm(quad[3] - quad[0])
                + np.linalg.norm(quad[2] - quad[1])
            )
            / 2
        )

    if width < 12 or height < 8 or not 0.7 <= width / height <= 40:
        return None

    target = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        np.float32,
    )
    # Порядок вершин задан относительно текста. Геометрическая сортировка
    # по левому верхнему углу изображения потеряла бы его направление.
    matrix = cv2.getPerspectiveTransform(quad, target)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return crop if crop.std() > 3 else None


def load_validation(cache: Path) -> tuple[list[dict], Path]:
    """Скачать аннотации и извлечь сцены из одного архива validation"""

    directory = cache / "hiertext_validation"
    annotation_path = download(
        ANNOTATIONS_URL,
        directory / "validation.jsonl.gz",
        progress=True,
    )

    print("Чтение аннотаций HierText validation", flush=True)

    with gzip.open(annotation_path, "rt", encoding="utf-8") as stream:
        annotations = json.load(stream)["annotations"]

    if not annotations:
        raise ValueError("В аннотациях HierText нет сцен")

    identifiers = [row["image_id"] for row in annotations]

    if (
        len(set(identifiers)) != len(identifiers)
        or any(not re.fullmatch(r"[0-9a-fA-F]+", value) for value in identifiers)
    ):
        raise ValueError("Некорректные или повторяющиеся ID сцен HierText")

    images = directory / "images"
    images.mkdir(parents=True, exist_ok=True)
    missing = set()

    for image_id in identifiers:
        path = images / f"{image_id}.jpg"

        if not path.is_file() or path.stat().st_size == 0:
            missing.add(image_id)

    if not missing:
        return annotations, images

    archive_path = download(
        ARCHIVE_URL,
        directory / "validation.tgz",
        progress=True,
    )

    with (
        tarfile.open(archive_path, "r:gz") as archive,
        tqdm(total=len(missing), desc="Извлечение сцен") as bar,
    ):
        for member in archive:
            name = Path(member.name)
            image_id = name.stem

            if (
                not member.isfile()
                or name.suffix.lower() != ".jpg"
                or image_id not in missing
            ):
                continue

            # Имя назначения берём из проверенного ID, а не из пути архива.
            destination = images / f"{image_id}.jpg"
            temporary = destination.with_name(destination.name + ".part")
            source = archive.extractfile(member)

            if source is None:
                raise ValueError(f"Не удалось открыть сцену в архиве: {image_id}")

            try:
                with source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)

                if member.size == 0 or temporary.stat().st_size != member.size:
                    raise ValueError(f"Сцена извлечена не полностью: {image_id}")

                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

            missing.remove(image_id)
            bar.update(1)

            if not missing:
                break

    if missing:
        raise ValueError(
            f"В архиве не найдены {len(missing)} сцен: {sorted(missing)[:5]}"
        )

    return annotations, images


def scene_candidates(annotation: dict) -> list[tuple[str, list, str, str]]:
    """Собрать кандидаты: ID внутри сцены, вершины, текст и вид кропа"""

    candidates = []

    for paragraph_index, paragraph in enumerate(annotation["paragraphs"]):
        for line_index, line in enumerate(paragraph["lines"]):
            if (
                not line.get("legible", False)
                or line.get("vertical", False)
                or line.get("handwritten", False)
            ):
                continue

            quad = line_quad(line)

            if quad is not None:
                candidates.append(
                    (
                        f"l{paragraph_index}_{line_index}",
                        quad.tolist(),
                        line["text"],
                        "line",
                    )
                )

            for word_index, word in enumerate(line["words"]):
                if not eligible_word(word):
                    continue

                candidates.append(
                    (
                        f"w{paragraph_index}_{line_index}_{word_index}",
                        word["vertices"],
                        word["text"],
                        "word",
                    )
                )

    return candidates


def process_scene(
    annotation: dict,
    image: NDArray[np.uint8],
    output: Path,
    max_crops: int,
    seed: int,
) -> dict:
    """Подготовить кропы одной сцены и её описание для поиска дубликатов"""

    image_id = annotation["image_id"]
    scene_id = "hier_" + image_id
    expected_shape = (
        annotation["image_height"],
        annotation["image_width"],
    )

    if image.shape[:2] != expected_shape:
        raise ValueError(f"Размер сцены не совпал с аннотацией: {scene_id}")

    scene = {
        "scene_id": scene_id,
        "source": "real",
        "dataset": "hiertext",
        "signature": image_signature(image),
    }

    candidates = scene_candidates(annotation)
    rng = np.random.default_rng(seed + int(image_id, 16))
    selected = []
    seen = set()
    skipped = Counter()

    for index in rng.permutation(len(candidates)):
        identifier, quad, text, kind = candidates[index]
        characters = re.sub(r"\W", "", text)

        # Консервативно исключаем надписи с неоднозначным направлением.
        if characters and all(char in "0OoОо8НHХXxхIil1" for char in characters):
            skipped["ambiguous_text"] += 1
            continue

        crop = upright_crop(image, quad)

        if crop is None:
            skipped["invalid_geometry_or_blank"] += 1
            continue

        key = pixel_key(crop)

        if key in seen:
            skipped["duplicate_in_scene"] += 1
            continue

        crop_id = f"{scene_id}_{identifier}"
        relative_path = f"crops/{crop_id}.png"

        save_crop(output / relative_path, crop)
        seen.add(key)

        selected.append(
            {
                "crop_id": crop_id,
                "path": relative_path,
                "scene_id": scene_id,
                "source": "real",
                "dataset": "hiertext",
                "kind": kind,
                "text": text,
                "script": script_group(text),
                "width": crop.shape[1],
                "height": crop.shape[0],
                "pixel_key": key,
                "orientation_method": "ordered_text_vertices",
                "applied_rotation": 0,
            }
        )

        if len(selected) >= max_crops:
            break

    return {
        "scene": scene,
        "crops": selected,
        "skipped": dict(skipped),
    }


def prepare_hiertext(
    output: Path,
    cache: Path,
    max_crops: int = 48,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Подготовить HierText и вернуть кропы, описания сцен и статистику

    Пути кропов относительны output. Изображения сохраняются в нормальной
    ориентации. Разделение данных и пары 0°/180° на этом шаге не создаются.
    Готовые записи сцен используются повторно при одинаковых настройках.
    """

    if max_crops < 1:
        raise ValueError("Максимальное число кропов должно быть положительным")

    if seed < 0:
        raise ValueError("Seed должен быть неотрицательным")

    metadata = output / "preparation_cache" / "hiertext"
    metadata.mkdir(parents=True, exist_ok=True)

    settings = {
        "annotation_commit": HIER_REVISION,
        "partition": "validation",
        "max_crops": max_crops,
        "seed": seed,
    }
    settings_path = metadata / "settings.json"

    if settings_path.is_file():
        saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))

        if saved_settings != settings:
            raise ValueError(
                "Настройки HierText изменились. Для пересборки данных "
                "запустите общую подготовку с --rebuild"
            )
    else:
        if any(metadata.glob("*.json")):
            raise ValueError(
                "Найдены записи HierText без настроек подготовки. "
                "Запустите общую подготовку с --rebuild"
            )

        save_json(settings_path, settings)

    annotations, images = load_validation(cache)
    annotations = sorted(annotations, key=lambda row: row["image_id"])

    records = []
    scenes = []
    skipped = Counter()
    scenes_with_crops = 0

    for annotation in tqdm(annotations, desc="Кропы HierText", unit="scene"):
        image_id = annotation["image_id"]
        saved = metadata / f"{image_id}.json"
        state = None

        if saved.is_file():
            state = json.loads(saved.read_text(encoding="utf-8"))

            if state["scene"]["scene_id"] != "hier_" + image_id:
                raise ValueError(f"Кэш содержит другую сцену: {saved}")

            for row in state["crops"]:
                path = output / row["path"]

                if not path.is_file() or path.stat().st_size == 0:
                    state = None
                    break

        if state is None:
            image = read_bgr(images / f"{image_id}.jpg")
            state = process_scene(annotation, image, output, max_crops, seed)

            # Запись появляется только после успешного сохранения всех PNG.
            save_json(saved, state)

        records.extend(state["crops"])
        scenes.append(state["scene"])
        skipped.update(state["skipped"])
        scenes_with_crops += bool(state["crops"])

    if not records:
        raise ValueError("Не удалось подготовить реальные кропы HierText")

    frame = pd.DataFrame(records).sort_values("crop_id").reset_index(drop=True)
    stats = {
        "dataset": "hiertext",
        "partition": "validation",
        "settings": settings,
        "archive_url": ARCHIVE_URL,
        "annotations_url": ANNOTATIONS_URL,
        "scenes_total": len(scenes),
        "scenes_with_crops": scenes_with_crops,
        "base_crops": len(frame),
        "word_crops": int((frame.kind == "word").sum()),
        "line_crops": int((frame.kind == "line").sum()),
        "skipped_before_scene_limit": dict(skipped),
    }

    print(
        f"HierText готов: {len(frame)} кропов из {scenes_with_crops} сцен",
        flush=True,
    )

    return frame, scenes, stats
