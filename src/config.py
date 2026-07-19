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
LEXICAL_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-zа-я0-9]+",
    re.IGNORECASE,
)
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

# Параметры переноса ответов похожих запросов
QUERY_WORD_WEIGHT: Final[float] = 0.3
QUERY_CHAR_WEIGHT: Final[float] = 0.7
MEMORY_NEIGHBORS: Final[int] = 30
MEMORY_POWER: Final[float] = 2.0
MEMORY_THRESHOLD: Final[float] = 0.0
MEMORY_FREQUENCY_POWER: Final[float] = 0.2
COOCCURRENCE_WEIGHT: Final[float] = 0.15

# Параметры классификации запросов
CLASSIFIER_CHAR_WEIGHT: Final[float] = 4.0
CLASSIFIER_C: Final[float] = 0.3
CLASSIFIER_MAX_ITERATIONS: Final[int] = 2000
CLASSIFIER_PRIOR_WEIGHT: Final[float] = 0.1
OOF_SPLITS: Final[int] = 5
OOF_SEEDS: Final[tuple[int, int, int]] = (7, 42, 2026)

# Параметры Qwen embeddings
EMBEDDING_MODEL: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION: Final[str] = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDING_ARTIFACT: Final[Path] = Path("artifacts/qwen3_calibration_embeddings.npz")
SUBMISSION_EMBEDDING_ARTIFACT: Final[Path] = Path(
    "artifacts/qwen3_submission_embeddings.npz"
)
MODEL_CACHE_DIR: Final[Path] = Path("models")
EMBEDDING_BATCH_SIZE: Final[int] = 4
EMBEDDING_MAX_LENGTH: Final[int] = 512
EMBEDDING_DIMENSION: Final[int] = 1024
EMBEDDING_PASSAGE_CHARACTERS: Final[int] = 6000
EMBEDDING_CACHE_VERSION: Final[int] = 1
DEFAULT_DEVICE: Final[str] = "cuda"
ARTICLE_QUERY_TASK: Final[str] = (
    "Given a Russian Avito customer support question, retrieve all Help Center "
    "articles required to resolve it. More than one article may be relevant."
)
MEMORY_QUERY_TASK: Final[str] = (
    "Given a Russian Avito customer support question, retrieve other support "
    "questions that require the same Help Center articles."
)

# Параметры Qwen query memory
DENSE_MEMORY_POWER: Final[float] = 3.0
DENSE_MEMORY_THRESHOLD: Final[float] = 0.4
DENSE_MEMORY_FREQUENCY_POWER: Final[float] = 0.25

# Веса быстрого гибридного поиска
DIRECT_LEXICAL_WEIGHT: Final[float] = 0.15
DIRECT_DENSE_WEIGHT: Final[float] = 0.85
FUSION_DIRECT_WEIGHT: Final[float] = 0.35
FUSION_DENSE_MEMORY_WEIGHT: Final[float] = 0.45
FUSION_LEXICAL_MEMORY_WEIGHT: Final[float] = 0.05
FUSION_CLASSIFIER_WEIGHT: Final[float] = 0.15

# Параметры Qwen reranker
RERANKER_MODEL: Final[str] = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_REVISION: Final[str] = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
RERANKER_ARTIFACT: Final[Path] = Path("artifacts/qwen3_reranker_calibration.npz")
SUBMISSION_RERANKER_ARTIFACT: Final[Path] = Path(
    "artifacts/qwen3_reranker_submission.npz"
)
RERANKER_BATCH_SIZE: Final[int] = 16
RERANKER_MAX_LENGTH: Final[int] = 256
RERANKER_CANDIDATES: Final[int] = 15
RERANKER_CHUNK_SIZE: Final[int] = 120
RERANKER_CHUNK_OVERLAP: Final[int] = 30
RERANKER_CHAR_MAX_FEATURES: Final[int] = 300_000
RERANKER_CACHE_VERSION: Final[int] = 1
RERANKER_TASK: Final[str] = (
    "Given a Russian Avito customer support question, determine whether this "
    "Help Center passage is required to resolve it. More than one passage may "
    "be relevant."
)

# Веса выбора фрагментов и финального ранжирования
CHUNK_WORD_WEIGHT: Final[float] = 0.4
CHUNK_CHAR_WEIGHT: Final[float] = 0.6
HIGH_SCORE_BASE_WEIGHT: Final[float] = 0.84
HIGH_SCORE_RERANKER_WEIGHT: Final[float] = 0.16

# Параметры запуска и результата
DEFAULT_OUTPUT_FILE: Final[Path] = Path("answer.csv")
FAST_MODE: Final[str] = "fast"
HIGH_SCORE_MODE: Final[str] = "high-score"
RUN_MODES: Final[tuple[str, str]] = (FAST_MODE, HIGH_SCORE_MODE)
