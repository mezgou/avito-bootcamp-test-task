"""Общие операции загрузки, чтения и сохранения внешних данных"""

import json
import math
import re
import shutil
import time
import zipfile
from datetime import UTC
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

_CHUNK_SIZE = 1024 * 1024
_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

# Паузы разделяются всеми потоками, обращающимися к одному серверу.
_REQUEST_LOCK = Lock()
_NEXT_REQUEST: dict[str, float] = {}


def _wait_for_request(host: str, interval: float) -> None:
    """Выдержать интервал между запросами к одному серверу"""

    while True:
        with _REQUEST_LOCK:
            now = time.monotonic()
            remaining = _NEXT_REQUEST.get(host, 0.0) - now

            if remaining <= 0:
                _NEXT_REQUEST[host] = now + interval
                return

        time.sleep(min(remaining, 30.0))


def _pause_requests(host: str, delay: float) -> None:
    """Отложить следующие запросы к серверу после ошибки"""

    with _REQUEST_LOCK:
        _NEXT_REQUEST[host] = max(
            _NEXT_REQUEST.get(host, 0.0),
            time.monotonic() + delay,
        )


def _retry_after(value: str | None, fallback: float) -> float:
    """Прочитать Retry-After как секунды или дату HTTP"""

    if value is None:
        return fallback

    try:
        seconds = float(value)
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)

            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)

            seconds = deadline.timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return fallback

    if not math.isfinite(seconds):
        return fallback

    return max(fallback, seconds)


def _download_file(
    url: str,
    temporary: Path,
    name: str,
    progress: bool,
    timeout: float,
) -> None:
    """Скачать файл во временный путь и проверить завершённость ответа"""

    request = Request(
        url,
        headers={
            "User-Agent": "avito-data-preparation",
            "Accept-Encoding": "identity",
        },
    )

    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"Неожиданный статус ответа: HTTP {response.status}")

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            raise ValueError("Вместо файла получена HTML-страница")

        size = response.headers.get("Content-Length")
        expected = int(size) if size is not None else None
        received = 0

        with (
            temporary.open("wb") as stream,
            tqdm(
                total=expected,
                desc=name,
                unit="B",
                unit_scale=True,
                disable=not progress,
            ) as bar,
        ):
            while chunk := response.read(_CHUNK_SIZE):
                stream.write(chunk)
                received += len(chunk)
                bar.update(len(chunk))

    if received == 0:
        raise ValueError("Получен пустой файл")

    if expected is not None and received != expected:
        raise ValueError(
            f"Неполная загрузка: получено {received} байт из {expected}"
        )


def download(
    url: str,
    path: Path,
    progress: bool = False,
    *,
    attempts: int = 5,
    timeout: float = 60.0,
    request_interval: float = 0.0,
) -> Path:
    """Скачать файл атомарно или вернуть уже готовый непустой файл

    Сетевые ошибки повторяются с паузой и сообщением в журнале.
    request_interval задаёт минимальную паузу между запросами к серверу.
    Готовый кэш повторно не скачивается. Незавершённая загрузка при повторе
    начинается сначала; докачка по HTTP Range не используется.
    """

    if attempts < 1:
        raise ValueError("Число попыток должно быть положительным")

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Таймаут должен быть положительным конечным числом")

    if not math.isfinite(request_interval) or request_interval < 0:
        raise ValueError("Интервал запросов должен быть неотрицательным")

    address = urlsplit(url)

    if address.scheme not in {"http", "https"} or not address.hostname:
        raise ValueError("Для загрузки нужен адрес HTTP или HTTPS")

    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    host = address.netloc.lower()

    for attempt in range(attempts):
        _wait_for_request(host, request_interval)

        try:
            _download_file(url, temporary, path.name, progress, timeout)
            temporary.replace(path)

            return path

        except HTTPError as error:
            code = error.code
            retry_after = error.headers.get("Retry-After")
            error.close()

            if code not in _RETRYABLE_CODES or attempt == attempts - 1:
                raise RuntimeError(
                    f"Не удалось скачать {path.name} с {host}: HTTP {code}"
                ) from error

            fallback = (
                min(60.0 * 2**attempt, 900.0)
                if code == 429
                else min(2.0**attempt, 30.0)
            )
            delay = _retry_after(retry_after, fallback)
            _pause_requests(host, delay)

            print(
                f"{path.name}: HTTP {code}; пауза не менее {delay:.0f} с; "
                f"следующая попытка {attempt + 2}/{attempts}",
                flush=True,
            )

        except (
            URLError,
            TimeoutError,
            ConnectionError,
            HTTPException,
            ValueError,
        ) as error:
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Не удалось скачать {path.name} с {host}: {error}"
                ) from error

            delay = min(2.0**attempt, 30.0)
            _pause_requests(host, delay)

            print(
                f"{path.name}: {type(error).__name__}: {error}; "
                f"пауза {delay:.0f} с; "
                f"следующая попытка {attempt + 2}/{attempts}",
                flush=True,
            )

        finally:
            temporary.unlink(missing_ok=True)

    raise RuntimeError(f"Загрузка не завершена: {path.name}")


def unpack_single(path: Path) -> Path:
    """Заменить ZIP его единственным файлом, не используя пути из архива"""

    if not zipfile.is_zipfile(path):
        return path

    temporary = path.with_name(path.name + ".part")

    try:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]

            if len(members) != 1:
                raise ValueError(f"Ожидался один файл в архиве: {path}")

            member = members[0]

            if not 0 < member.file_size <= 256 * 1024 * 1024:
                raise ValueError(f"Неожиданный размер файла в архиве: {path}")

            with (
                archive.open(member) as source,
                temporary.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination, length=_CHUNK_SIZE)

        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return path


def read_bgr(path: Path) -> NDArray[np.uint8]:
    """Прочитать цветное изображение в формате BGR"""

    encoded = np.fromfile(path, dtype=np.uint8)

    if encoded.size == 0:
        raise ValueError(f"Файл изображения пуст: {path}")

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    return image


def save_crop(path: Path, image: NDArray[np.uint8]) -> None:
    """Атомарно сохранить BGR-кроп в PNG без потерь"""

    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ValueError("Ожидалось непустое BGR-изображение uint8 формата HWC")

    if path.suffix.lower() != ".png":
        raise ValueError(f"Кроп должен сохраняться в файл PNG: {path}")

    ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )

    if not ok:
        raise ValueError(f"Не удалось закодировать кроп: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")

    try:
        encoded.tofile(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, value: object) -> None:
    """Атомарно сохранить JSON в UTF-8"""

    content = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")

    try:
        temporary.write_text(content + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_text(text: str) -> str:
    """Оставить буквы и цифры без различия регистра"""

    return "".join(character for character in text.casefold() if character.isalnum())


def script_group(text: str) -> str:
    """Определить наличие русской кириллицы, латиницы или цифр"""

    cyrillic = bool(re.search(r"[А-Яа-яЁё]", text))
    latin = bool(re.search(r"[A-Za-z]", text))

    if cyrillic and latin:
        return "mixed"

    if cyrillic:
        return "cyrillic"

    if latin:
        return "latin"

    if normalize_text(text).isdigit():
        return "digits"

    return "other"
