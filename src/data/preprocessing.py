"""Чтение и подготовка текстовых изображений"""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# Средние и стандартные отклонения каналов RGB для нормализации ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_rgb(path: str | Path) -> NDArray[np.uint8]:
    """Прочитать изображение в формате RGB"""

    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(
    image: NDArray[np.uint8],
    size: tuple[int, int] = (160, 80),
) -> NDArray[np.float32]:
    """Подготовить RGB-кроп в формате CHW; size - ширина и высота"""

    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ValueError("Ожидалось непустое RGB-изображение uint8 формата HWC")

    resized = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_LINEAR,
    )

    normalized = resized.astype(np.float32) * np.float32(1 / 255)
    normalized = (normalized - MEAN) / STD

    return np.ascontiguousarray(normalized.transpose(2, 0, 1))
