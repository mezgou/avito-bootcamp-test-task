"""Проверка направления внешнего текста по его транскрипции"""

import math
from pathlib import Path

import cv2
import numpy as np
import paddle
import yaml
from numpy.typing import NDArray
from paddle.inference import Config, create_predictor

from src.data.sources import download

MODEL_REVISION = "712d2d65556ccc1ea7b5d2bb232b018838b6a3ab"
MODEL_URL = (
    "https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec/resolve/"
    + MODEL_REVISION
)


class TextRecognizer:
    """Небольшой CTC-распознаватель для подготовки RusTitW"""

    def __init__(self, directory: Path, device: str = "cpu") -> None:
        """Загрузить распознаватель и настроить устройство"""

        for name in ("inference.yml", "inference.json", "inference.pdiparams"):
            download(f"{MODEL_URL}/{name}", directory / name)

        settings = yaml.safe_load(
            (directory / "inference.yml").read_text(encoding="utf-8")
        )
        operations = settings["PreProcess"]["transform_ops"]

        decode = next(
            op["DecodeImage"] for op in operations if "DecodeImage" in op
        )
        resize = next(
            op["RecResizeImg"] for op in operations if "RecResizeImg" in op
        )

        if decode["img_mode"] != "BGR" or resize["image_shape"] != [
            3,
            48,
            320,
        ]:
            raise ValueError(
                "Изменилась официальная обработка входа распознавателя"
            )

        if settings["PostProcess"]["name"] != "CTCLabelDecode":
            raise ValueError("Ожидался выход CTC")

        self.characters = (
            [""] + settings["PostProcess"]["character_dict"] + [" "]
        )
        self.device = device

        config = Config(
            str(directory / "inference.json"),
            str(directory / "inference.pdiparams"),
        )

        if device == "cpu":
            config.disable_gpu()
            config.disable_mkldnn()
            config.set_cpu_math_library_num_threads(2)
        elif device.startswith("gpu:"):
            index = int(device.split(":")[1])

            if (
                not paddle.is_compiled_with_cuda()
                or not 0 <= index < paddle.device.cuda.device_count()
            ):
                raise RuntimeError(
                    f"Недоступно запрошенное устройство: {device}"
                )

            paddle.device.set_device(device)
            config.enable_use_gpu(256, index)

            if not config.use_gpu():
                raise RuntimeError("Paddle Inference не включил GPU")
        else:
            raise ValueError("Используйте DEVICE=cpu или DEVICE=gpu:0")

        config.disable_glog_info()
        self.predictor = create_predictor(config)

        inputs, outputs = (
            self.predictor.get_input_names(),
            self.predictor.get_output_names(),
        )

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Неожиданное число входов или выходов OCR")

        self.input = self.predictor.get_input_handle(inputs[0])
        self.output = self.predictor.get_output_handle(outputs[0])

    def recognize(
        self,
        images: list[NDArray[np.uint8]],
        batch_size: int = 16,
    ) -> list[str]:
        """Распознать BGR-кропы с группировкой по ширине"""

        if batch_size < 1:
            raise ValueError("Размер пачки должен быть положительным")

        result = [""] * len(images)
        order = sorted(
            range(len(images)),
            key=lambda i: images[i].shape[1] / images[i].shape[0],
        )

        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            ratios = [images[i].shape[1] / images[i].shape[0] for i in indices]
            width = min(3200, max(320, int(48 * max(ratios))))
            batch = np.zeros((len(indices), 3, 48, width), dtype=np.float32)

            for position, index in enumerate(indices):
                image = images[index]
                resized_width = min(
                    width, math.ceil(48 * image.shape[1] / image.shape[0])
                )

                resized = cv2.resize(image, (resized_width, 48)).astype(
                    np.float32
                )
                normalized = (
                    resized.transpose(2, 0, 1) / np.float32(255) - 0.5
                ) / 0.5
                batch[position, :, :, :resized_width] = normalized

            self.input.reshape(batch.shape)
            self.input.copy_from_cpu(batch)
            self.predictor.run()
            output = self.output.copy_to_cpu()

            if (
                output.ndim != 3
                or output.shape[0] != len(indices)
                or output.shape[2] != len(self.characters)
                or not np.isfinite(output).all()
            ):
                raise ValueError(f"Некорректный выход OCR: {output.shape}")

            for index, probabilities in zip(indices, output, strict=True):
                tokens = probabilities.argmax(axis=1)
                keep = np.r_[True, tokens[1:] != tokens[:-1]] & (tokens != 0)
                result[index] = "".join(
                    self.characters[int(token)] for token in tokens[keep]
                )

        return result
