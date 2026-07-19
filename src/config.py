import re
from pathlib import Path
from typing import Final

# Пути и имена файлов с исходными данными
DEFAULT_DATA_DIR: Final[Path] = Path("data/candidate_public/candidate_data")
ARTICLES_FILE: Final[str] = "articles.f"
CALIBRATION_FILE: Final[str] = "calibration.f"
TEST_FILE: Final[str] = "test.f"

# Обязательные колонки входных таблиц
ARTICLE_COLUMNS: Final[tuple[str, str, str]] = ("article_id", "title", "body")
CALIBRATION_COLUMNS: Final[tuple[str, str, str]] = (
    "query_id",
    "query_text",
    "ground_truth",
)
TEST_COLUMNS: Final[tuple[str, str]] = ("query_id", "query_text")

# Правила очистки и нормализации текста
REMOVED_TAGS: Final[tuple[str, str, str, str]] = (
    "input",
    "noscript",
    "script",
    "style",
)
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
TEXT_TRANSLATION: Final[dict[int, int | str | None]] = str.maketrans(
    {"ё": "е", "Ё": "Е"}
)

# Размер и перекрытие текстовых фрагментов
CHUNK_SIZE: Final[int] = 150
CHUNK_OVERLAP: Final[int] = 35
MAX_CHUNK_SIZE: Final[int] = 180
