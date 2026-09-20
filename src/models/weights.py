"""Скачивание исходных весов классификатора ориентации строк"""

from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

PRETRAINED_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/"
    "paddlex/official_pretrained_model/"
    "PP-LCNet_x1_0_textline_ori_pretrained.pdparams"
)


def download_pretrained(
    path: str | Path = "weights/pretrained.pdparams",
) -> Path:
    """Скачать официальные веса, если локального файла ещё нет"""

    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")

    try:
        with (
            urlopen(PRETRAINED_URL, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            expected_size = response.headers.get("Content-Length")
            copyfileobj(response, output)

        actual_size = temporary.stat().st_size
        if actual_size == 0 or (
            expected_size is not None and actual_size != int(expected_size)
        ):
            raise OSError("Файл весов скачан не полностью")

        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return path
