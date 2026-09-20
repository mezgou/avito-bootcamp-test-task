"""Загрузка модели и получение вероятности поворота на 180 градусов"""

from pathlib import Path

import numpy as np
import paddle
from numpy.typing import NDArray

from src.models.lcnet import PPLCNet


def load_model(weights: str | Path, device: str = "cpu") -> PPLCNet:
    """Загрузить обучаемые веса из файла pdparams"""

    paddle.device.set_device(device)
    model = PPLCNet()
    state = paddle.load(str(weights), return_numpy=True)
    expected = model.state_dict()

    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            f"Несовместимые веса: отсутствуют {missing}, лишние {unexpected}"
        )

    for name, parameter in expected.items():
        if tuple(parameter.shape) != tuple(state[name].shape):
            raise ValueError(f"Не совпадает размер параметра {name}")

    model.set_state_dict(state)
    model.eval()
    return model


@paddle.no_grad()
def predict_batch(
    model: PPLCNet,
    batch: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Вернуть p_180 для подготовленной пачки изображений NCHW"""

    if (
        batch.ndim != 4
        or batch.shape[1] != 3
        or batch.dtype != np.float32
        or batch.size == 0
    ):
        raise ValueError("Ожидалась непустая пачка float32 формата NCHW")

    model.eval()
    inputs = paddle.to_tensor(batch, place=model.parameters()[0].place)
    logits = model(inputs)
    probabilities = paddle.nn.functional.softmax(logits, axis=1)
    p_180 = probabilities[:, 1].numpy().astype(np.float64)

    if not np.isfinite(p_180).all():
        raise ValueError("Модель вернула некорректные вероятности")

    return p_180
