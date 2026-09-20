"""Объединение дубликатов и воспроизводимое разделение групп сцен"""

import hashlib
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

SPLITS = ("train", "development", "calibration", "holdout")
SPLIT_FRACTIONS = (0.7, 0.1, 0.1, 0.1)
PHASH_BLOCKS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))


def pixel_key(image: NDArray[np.uint8]) -> str:
    """Получить ключ точных пикселей, одинаковый для поворотов 0° и 180°"""

    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ValueError("Ожидалось непустое BGR-изображение uint8 формата HWC")

    views = (
        np.ascontiguousarray(image),
        np.ascontiguousarray(image[::-1, ::-1]),
    )
    digest = min(
        hashlib.blake2b(view.tobytes(), digest_size=16).hexdigest()
        for view in views
    )

    return f"{image.shape[1]}x{image.shape[0]}:{digest}"


def image_signature(image: NDArray[np.uint8]) -> dict:
    """Описать сцену для поиска точных и визуально близких копий"""

    exact = pixel_key(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    hashes = []

    for view in (small, small[::-1, ::-1]):
        frequencies = cv2.dct(view.astype(np.float32))[:8, :8].flatten()
        bits = frequencies > np.median(frequencies[1:])
        bits[0] = False

        value = int.from_bytes(np.packbits(bits).tobytes(), "big")
        hashes.append(value)

    return {
        "exact": exact,
        "phash": hashes,
        "aspect": image.shape[1] / image.shape[0],
        "gray_std": float(gray.std()),
    }


def assign_splits(
    crops: pd.DataFrame,
    scenes: list[dict],
    seed: int = 42,
    previous: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Вернуть manifest, таблицу сцен и статистику разделения

    Пропорции 70/10/10/10 задаются по группам, а не по числу кропов.
    Дубликаты объединяются до разделения. Все кропы одной сцены остаются
    вместе; синтетические варианты одного текста уже имеют общий scene_id.

    previous — существующий manifest при расширении корпуса. Его части
    сохраняются. Конфликт при объединении ранее разных частей вызывает
    ошибку. Функция не читает изображения и не изменяет файлы на диске.
    """

    if seed < 0:
        raise ValueError("Seed должен быть неотрицательным")

    if crops.empty or not scenes:
        raise ValueError("Нет кропов или сцен для разделения")

    required = {
        "crop_id", "path", "scene_id", "source", "dataset", "script", "pixel_key",
    }
    missing = required - set(crops.columns)

    if missing:
        raise ValueError(f"В таблице кропов нет столбцов: {sorted(missing)}")

    if crops[list(required)].isna().any().any():
        raise ValueError("Обязательные поля кропов содержат пропуски")

    if crops.crop_id.duplicated().any() or crops.path.duplicated().any():
        raise ValueError("Повторяющиеся crop_id или пути до удаления дубликатов")

    for column in ("crop_id", "path", "scene_id", "pixel_key"):
        if not crops[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"Столбец {column} должен содержать непустые строки")

    scenes = sorted(scenes, key=lambda scene: scene["scene_id"])
    scene_ids = [scene["scene_id"] for scene in scenes]

    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Повторяющиеся scene_id в описаниях сцен")

    unknown = set(crops.scene_id) - set(scene_ids)

    if unknown:
        raise ValueError(f"Кропы ссылаются на неизвестные сцены: {sorted(unknown)[:5]}")

    for column in ("source", "dataset"):
        expected = {scene["scene_id"]: scene[column] for scene in scenes}

        if not crops[column].eq(crops.scene_id.map(expected)).all():
            raise ValueError(f"Описание сцен и кропов расходится по {column}")

    crops = crops.sort_values("crop_id").reset_index(drop=True)
    has_previous = previous is not None and not previous.empty

    if has_previous:
        previous_required = {"crop_id", "scene_id", "group_id", "split"}

        if not previous_required.issubset(previous.columns):
            raise ValueError("В предыдущем manifest нет ID, групп или частей данных")

        if previous[list(previous_required)].isna().any().any():
            raise ValueError("Предыдущий manifest содержит пропуски в ID или split")

        if previous.crop_id.duplicated().any():
            raise ValueError("В предыдущем manifest повторяются crop_id")

        if not previous.split.isin(SPLITS).all():
            raise ValueError("В предыдущем manifest неизвестные части данных")

        for column in ("scene_id", "group_id"):
            if previous.groupby(column).split.nunique().max() != 1:
                raise ValueError(f"Предыдущий manifest содержит пересечение по {column}")

        current = crops.set_index("crop_id")
        missing_ids = set(previous.crop_id) - set(current.index)

        if missing_ids:
            raise ValueError(
                f"При расширении пропали кропы: {sorted(missing_ids)[:5]}. "
                "Изменение состава требует явной пересборки через --rebuild"
            )

        stable_columns = ["scene_id"]

        if "pixel_key" in previous.columns:
            stable_columns.append("pixel_key")

        for column in stable_columns:
            actual = current.loc[previous.crop_id, column].to_numpy()

            if not np.array_equal(actual, previous[column].to_numpy()):
                raise ValueError(
                    f"Изменились исходные кропы по {column}; нужен --rebuild"
                )

    parents = {scene_id: scene_id for scene_id in scene_ids}

    def root(item: str) -> str:
        """Найти корень группы и сократить путь следующих обращений"""

        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]

        return item

    def unite(first: str, second: str) -> None:
        """Объединить группы с устойчивым выбором корня"""

        left, right = sorted((root(first), root(second)))
        parents[right] = left

    # Сохраняем ранее установленные связи сцен при расширении корпуса.
    if has_previous:
        for _, rows in previous.groupby("group_id", sort=True):
            members = sorted(rows.scene_id.unique())

            for member in members[1:]:
                unite(members[0], member)

    exact_scenes: dict[str, str] = {}
    exact_crops: dict[str, str] = {}
    buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    exact_links = 0
    near_links = 0
    crop_links = 0

    for index, scene in enumerate(scenes):
        signature = scene["signature"]
        key = signature["exact"]

        if key in exact_scenes:
            unite(scene["scene_id"], exact_scenes[key])
            exact_links += 1
        else:
            exact_scenes[key] = scene["scene_id"]

        # Пустой pHash у синтетики отключает поиск близких сцен для неё.
        if not signature["phash"] or signature["gray_std"] < 8:
            continue

        candidates = set()

        # Если различаются не более четырёх битов, хотя бы один из пяти
        # непересекающихся блоков совпадает полностью.
        for value in signature["phash"]:
            for block, (offset, size) in enumerate(PHASH_BLOCKS):
                block_value = (int(value) >> offset) & ((1 << size) - 1)
                candidates.update(buckets[(block, block_value)])

        for other_index in sorted(candidates):
            other = scenes[other_index]
            other_signature = other["signature"]
            aspect_ratio = signature["aspect"] / other_signature["aspect"]

            if not 0.95 <= aspect_ratio <= 1.05:
                continue

            distance = min(
                (int(first) ^ int(second)).bit_count()
                for first in signature["phash"]
                for second in other_signature["phash"]
            )

            if distance <= 4:
                unite(scene["scene_id"], other["scene_id"])
                near_links += 1

        for value in signature["phash"]:
            for block, (offset, size) in enumerate(PHASH_BLOCKS):
                block_value = (int(value) >> offset) & ((1 << size) - 1)
                buckets[(block, block_value)].add(index)

    # Совпадающий кроп связывает целые сцены, включая все их слова и строки.
    for row in crops.itertuples(index=False):
        if row.pixel_key in exact_crops:
            unite(row.scene_id, exact_crops[row.pixel_key])
            crop_links += 1
        else:
            exact_crops[row.pixel_key] = row.scene_id

    assignments = {}

    if has_previous:
        anchors: dict[str, set[str]] = defaultdict(set)
        old_scenes = previous[["scene_id", "split"]].drop_duplicates()

        for row in old_scenes.itertuples(index=False):
            anchors[root(row.scene_id)].add(row.split)

        conflicts = {
            group: sorted(parts)
            for group, parts in anchors.items()
            if len(parts) > 1
        }

        if conflicts:
            examples = dict(list(conflicts.items())[:3])

            raise ValueError(
                "Дубликаты связали ранее разные части данных: "
                f"{examples}. Сохранить прежний split невозможно. "
                "Проверьте конфликтующие группы перед пересборкой"
            )

        assignments = {
            group: next(iter(parts))
            for group, parts in anchors.items()
        }

    result = crops.copy()
    result["group_id"] = result.scene_id.map(root)

    # При совпадении пикселей отдаём приоритет уже принятому crop_id.
    if has_previous:
        result["_existing"] = result.crop_id.isin(previous.crop_id)
        result = result.sort_values(
            ["_existing", "crop_id"],
            ascending=[False, True],
        )

    result = result.drop_duplicates("pixel_key")
    result = result.drop(columns="_existing", errors="ignore")

    strata: dict[tuple, list[str]] = defaultdict(list)

    for group_id, rows in result.groupby("group_id", sort=True):
        key = (
            tuple(sorted(rows.dataset.unique())),
            bool(rows.script.isin(["cyrillic", "mixed"]).any()),
        )
        strata[key].append(group_id)

    for key, all_groups in sorted(strata.items()):
        groups = sorted(
            group for group in all_groups
            if group not in assignments
        )

        # Независимый seed страты исключает влияние порядка источников.
        digest = hashlib.blake2b(
            f"{seed}:{key!r}".encode(),
            digest_size=8,
        ).digest()
        stratum_seed = int.from_bytes(digest, "big")
        rng = np.random.default_rng(stratum_seed)
        rng.shuffle(groups)

        count = len(groups)
        boundaries = [0]
        boundaries.extend(
            round(float(fraction) * count)
            for fraction in np.cumsum(SPLIT_FRACTIONS)[:-1]
        )
        boundaries.append(count)

        for split, left, right in zip(
            SPLITS,
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            for group in groups[left:right]:
                assignments[group] = split

    result["split"] = result.group_id.map(assignments)
    result = result.sort_values("crop_id").reset_index(drop=True)

    if result.split.isna().any():
        raise ValueError("Не всем группам назначена часть данных")

    for column in ("scene_id", "group_id", "pixel_key"):
        if result.groupby(column).split.nunique().max() != 1:
            raise ValueError(f"Пересечение частей по {column}")

    groups = pd.DataFrame(
        [
            {
                "scene_id": scene["scene_id"],
                "source": scene["source"],
                "dataset": scene["dataset"],
                "group_id": root(scene["scene_id"]),
                "split": assignments.get(root(scene["scene_id"]), "excluded"),
            }
            for scene in scenes
        ]
    )

    stats = {
        "seed": seed,
        "fractions_by_group": dict(zip(SPLITS, SPLIT_FRACTIONS, strict=True)),
        "exact_scene_links": exact_links,
        "near_scene_links": near_links,
        "exact_crop_links": crop_links,
        "crops_before_deduplication": len(crops),
        "removed_duplicate_crops": len(crops) - len(result),
        "base_crops": len(result),
        "preserved_previous_assignments": has_previous,
        "largest_group_scenes": int(
            groups.groupby("group_id").scene_id.nunique().max()
        ),
        "crops_by_split": {
            split: int((result.split == split).sum())
            for split in SPLITS
        },
        "groups_by_split": {
            split: int(result.loc[result.split == split, "group_id"].nunique())
            for split in SPLITS
        },
        "deduplication_limit": (
            "pHash не обнаруживает все кадрирования и сложные дубликаты; "
            "возможны ошибочные объединения визуально похожих сцен"
        ),
    }

    return result, groups, stats
