"""Подготовка синтетических строк с кириллицей и латиницей"""

import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm.auto import tqdm

from src.data.sources import (
    download,
    read_bgr,
    save_crop,
    save_json,
    script_group,
)
from src.data.split import pixel_key

RUSSIAN_WORDS = ["Продажа", "квартиры", "доставка", "магазин", "мебель", "телефон", "ремонт", "инструмент", "одежда", "техника", "новый", "красивый", "недорого", "Москва", "Санкт-Петербург", "объявление", "велосипед", "работа", "услуги", "документы", "компьютер", "монитор", "сегодня", "скидка", "свежие", "продукты", "запчасти", "детские", "игрушки", "цена", "площадь"]

LATIN_WORDS = ["Delivery", "house", "apartment", "shop", "tools", "sale", "bicycle", "service", "computer", "fresh", "new", "quality", "repair", "today", "price", "screen", "electronics", "clothes", "garden", "winter", "summer", "books", "music", "camera"]

FONT_NAMES = (
    "DejaVuSans.ttf",
    "DejaVuSansMono.ttf",
    "DejaVuSerif.ttf",
)

FONT_ARCHIVE_URLS = (
    ("https://github.com/dejavu-fonts/dejavu-fonts/releases/download/"
    "version_2_37/dejavu-fonts-ttf-2.37.zip"),
    ("https://downloads.sourceforge.net/project/dejavu/dejavu/2.37/"
    "dejavu-fonts-ttf-2.37.zip"),
)

# Контрольная сумма официального архива: dejavu-fonts.github.io/Download.html.
FONT_ARCHIVE_SHA256 = (
    "7576310b219e04159d35ff61dd4a4ec4cdba4f35c00e002a136f00e96a908b0a"
)


def load_fonts(directory: Path = Path("data/fonts")) -> list[Path]:
    """Скачать DejaVu и лицензию или использовать готовые локальные файлы"""

    fonts = [directory / name for name in FONT_NAMES]
    license_path = directory / "LICENSE_DEJAVU.txt"
    required = [*fonts, license_path]

    if all(path.is_file() and path.stat().st_size > 0 for path in required):
        try:
            for path in fonts:
                ImageFont.truetype(str(path), 16)
        except OSError:
            print("Восстановление повреждённых файлов DejaVu", flush=True)
        else:
            return fonts

    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "dejavu-fonts-ttf-2.37.zip"

    for index, url in enumerate(FONT_ARCHIVE_URLS):
        try:
            download(url, archive_path, progress=True)
            break
        except RuntimeError:
            if index == len(FONT_ARCHIVE_URLS) - 1:
                raise

            print("Архив DejaVu недоступен; пробуем второе зеркало", flush=True)

    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    if checksum != FONT_ARCHIVE_SHA256:
        archive_path.unlink()

        raise ValueError(
            "Контрольная сумма архива DejaVu не совпала. "
            "Неподходящий архив удалён; повторите подготовку"
        )

    targets = {name: directory / name for name in FONT_NAMES}
    targets["LICENSE"] = license_path

    with zipfile.ZipFile(archive_path) as archive:
        members = {
            Path(item.filename).name: item
            for item in archive.infolist()
            if not item.is_dir()
        }

        for name, destination in targets.items():
            if name not in members:
                raise ValueError(f"В архиве DejaVu отсутствует файл: {name}")

            temporary = destination.with_name(destination.name + ".part")

            try:
                temporary.write_bytes(archive.read(members[name]))
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

    for path in fonts:
        ImageFont.truetype(str(path), 16)

    return fonts


def generate_text(rng: np.random.Generator, cyrillic: bool) -> str:
    """Составить строку из слов, необязательного числа и варианта регистра"""

    corpus = RUSSIAN_WORDS if cyrillic else LATIN_WORDS
    word_count = int(rng.integers(1, 7))
    text = " ".join(rng.choice(corpus, size=word_count))

    if rng.random() < 0.35:
        text += f" {int(rng.integers(10, 10000))}"

    if rng.random() < 0.35:
        text = text.upper()

    return text


def render_text(
    text: str,
    rng: np.random.Generator,
    fonts: list[Path],
) -> NDArray[np.uint8]:
    """Нарисовать строку в нормальной ориентации и добавить искажения"""

    font_path = fonts[int(rng.integers(len(fonts)))]
    font_size = int(rng.integers(16, 60))
    font = ImageFont.truetype(str(font_path), font_size)

    box = font.getbbox(text)
    padding = int(rng.integers(1, 8))
    width = box[2] - box[0] + 2 * padding
    height = box[3] - box[1] + 2 * padding

    background = int(rng.integers(165, 256))
    foreground = int(rng.integers(0, 100))

    if rng.random() < 0.25:
        background, foreground = foreground, background

    image = Image.new("RGB", (width, height), (background,) * 3)

    ImageDraw.Draw(image).text(
        (padding - box[0], padding - box[1]),
        text,
        font=font,
        fill=(foreground,) * 3,
    )

    target_height = int(rng.integers(10, 65))
    target_width = max(12, round(width * target_height / height))

    image = image.resize(
        (target_width, target_height),
        Image.Resampling.BILINEAR,
    )

    if rng.random() < 0.4:
        radius = float(rng.uniform(0.2, 0.9))
        image = image.filter(ImageFilter.GaussianBlur(radius))

    pixels = np.asarray(image).astype(np.float32)
    noise_std = float(rng.uniform(0, 5))
    pixels += rng.normal(0, noise_std, pixels.shape)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

    if rng.random() < 0.45:
        quality = int(rng.integers(35, 90))

        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )

        if not ok:
            raise ValueError("Не удалось закодировать JPEG для искажения")

        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if bgr is None:
            raise ValueError("Не удалось декодировать JPEG после искажения")

    return np.ascontiguousarray(bgr)


def prepare_synthetic(
    output: Path,
    count: int = 4000,
    seed: int = 42,
    fonts_dir: Path = Path("data/fonts"),
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Подготовить синтетику и вернуть кропы, описания сцен и статистику

    Каждые четыре строки содержат три кириллические и одну латинскую.
    Пути кропов относительны output. Изображения сохраняются в нормальной
    ориентации, а варианты одинакового текста получают общий scene_id.
    Разделение данных и пары 0°/180° создаются на следующих этапах.
    Шрифты и лицензия автоматически загружаются в fonts_dir.
    """

    if count < 1:
        raise ValueError("Число синтетических кропов должно быть положительным")

    if seed < 0:
        raise ValueError("Seed должен быть неотрицательным")

    fonts = load_fonts(fonts_dir)
    metadata = output / "preparation_cache" / "synthetic"
    metadata.mkdir(parents=True, exist_ok=True)

    settings = {
        "count": count,
        "seed": seed,
        "cyrillic_per_four": 3,
        "fonts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in fonts
        },
        "russian_words": RUSSIAN_WORDS,
        "latin_words": LATIN_WORDS,
    }
    settings_path = metadata / "settings.json"

    if settings_path.is_file():
        saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))

        if saved_settings != settings:
            raise ValueError(
                "Настройки синтетики изменились. Для пересборки данных "
                "запустите общую подготовку с --rebuild"
            )
    else:
        if any((output / "crops").glob("synth_*.png")):
            raise ValueError(
                "Найдены синтетические кропы без настроек подготовки. "
                "Запустите общую подготовку с --rebuild"
            )

        save_json(settings_path, settings)

    records = []
    scenes = []
    text_scenes: dict[str, str] = {}

    for number in tqdm(range(count), desc="Синтетические строки", unit="crop"):
        # Отдельный генератор для каждого кропа сохраняет результат при
        # продолжении прерванного запуска, независимо от готовых соседей.
        rng = np.random.default_rng(seed + number)
        cyrillic = number % 4 < 3
        text = generate_text(rng, cyrillic)

        crop_id = f"synth_{number:06d}"
        relative_path = f"crops/{crop_id}.png"
        path = output / relative_path

        normalized_text = " ".join(text.casefold().split())
        first_variant = normalized_text not in text_scenes
        scene_id = text_scenes.setdefault(normalized_text, crop_id)

        bgr = None

        if path.is_file() and path.stat().st_size > 0:
            try:
                bgr = read_bgr(path)
            except (ValueError, cv2.error):
                tqdm.write(f"Повторная генерация повреждённого PNG: {crop_id}")

        if bgr is None:
            bgr = render_text(text, rng, fonts)
            save_crop(path, bgr)

        key = pixel_key(bgr)

        records.append(
            {
                "crop_id": crop_id,
                "path": relative_path,
                "scene_id": scene_id,
                "source": "synthetic",
                "dataset": "synthetic",
                "kind": "line",
                "text": text,
                "script": script_group(text),
                "width": bgr.shape[1],
                "height": bgr.shape[0],
                "pixel_key": key,
                "orientation_method": "rendered_text",
                "applied_rotation": 0,
            }
        )

        if first_variant:
            # Группируем синтетику по тексту. pHash отключён, чтобы похожие
            # фон и шрифт не объединяли разные надписи в большие группы.
            scenes.append(
                {
                    "scene_id": scene_id,
                    "source": "synthetic",
                    "dataset": "synthetic",
                    "signature": {
                        "exact": key,
                        "phash": [],
                        "aspect": bgr.shape[1] / bgr.shape[0],
                        "gray_std": float(
                            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std()
                        ),
                    },
                }
            )

    frame = pd.DataFrame(records).sort_values("crop_id").reset_index(drop=True)
    stats = {
        "dataset": "synthetic",
        "font_directory": fonts_dir.as_posix(),
        "font_archive_sources": list(FONT_ARCHIVE_URLS),
        "font_archive_sha256": FONT_ARCHIVE_SHA256,
        "settings": settings,
        "base_crops": len(frame),
        "scenes_total": len(scenes),
        "cyrillic_crops": int((frame.script == "cyrillic").sum()),
        "latin_crops": int((frame.script == "latin").sum()),
        "grouping": "Одинаковые тексты без учёта регистра находятся вместе",
    }

    print(
        f"Синтетика готова: {len(frame)} кропов; "
        f"кириллица — {stats['cyrillic_crops']}, "
        f"латиница — {stats['latin_crops']}",
        flush=True,
    )

    return frame, scenes, stats
