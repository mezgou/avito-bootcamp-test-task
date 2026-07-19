import re
from pathlib import Path
from typing import Final, Literal

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

# Параметры метрики ранжирования
METRIC_CUTOFF: Final[int] = 10

# Параметры TF-IDF поиска
WORD_ANALYZER: Final[Literal["word"]] = "word"
CHAR_ANALYZER: Final[Literal["char_wb"]] = "char_wb"
WORD_NGRAM_RANGE: Final[tuple[int, int]] = (1, 2)
CHAR_NGRAM_RANGE: Final[tuple[int, int]] = (3, 5)
WORD_MIN_DF: Final[int] = 1
BODY_CHAR_MIN_DF: Final[int] = 2
TITLE_CHAR_MIN_DF: Final[int] = 1
WORD_TOKEN_PATTERN: Final[str] = r"(?u)\b\w\w+\b"
WORD_SUBLINEAR_TF: Final[bool] = False
CHAR_SUBLINEAR_TF: Final[bool] = True
SCORE_EPSILON: Final[float] = 1e-8
