import unicodedata
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import pyarrow as pa
from bs4 import BeautifulSoup

from config import (
    ARTICLE_COLUMNS,
    ARTICLES_FILE,
    CALIBRATION_COLUMNS,
    CALIBRATION_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_DATA_DIR,
    MAX_CHUNK_SIZE,
    REMOVED_TAGS,
    TEST_COLUMNS,
    TEST_FILE,
    TEXT_TRANSLATION,
    WHITESPACE_PATTERN,
)


def normalize_text(value: str) -> str:
    """Приводит текст к единому формату"""
    normalized = unicodedata.normalize("NFKC", value).translate(TEXT_TRANSLATION)
    normalized = "".join(
        character
        for character in normalized
        if character.isprintable() or character.isspace()
    )

    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def html_to_text(value: str) -> str:
    """Извлекает полезный текст из HTML-разметки статьи"""
    soup = BeautifulSoup(value, "lxml")

    for element in soup.find_all(REMOVED_TAGS):
        element.decompose()

    for headline in soup.find_all("headline"):
        if headline.get_text(" ", strip=True):
            continue

        name = headline.get("name")
        if isinstance(name, str) and name.strip():
            headline.replace_with(f" {normalize_text(name)} ")

    for chunk in soup.find_all("chunk"):
        title = chunk.get("title")
        if isinstance(title, str) and title.strip():
            chunk.insert(0, f" {normalize_text(title)}. ")

    for cell in soup.find_all(("th", "td")):
        cell.append(" | ")

    for image in soup.find_all("img"):
        alt = image.get("alt")
        replacement = normalize_text(unquote(alt)) if isinstance(alt, str) else ""

        if replacement:
            image.replace_with(f" {replacement} ")
        else:
            image.decompose()

    return normalize_text(soup.get_text(" ", strip=True))


def parse_ground_truth(value: str) -> tuple[int, ...]:
    """Преобразует строку эталонных article_id в кортеж чисел"""
    return tuple(int(token) for token in value.split())


def _read_feather(path: Path) -> pd.DataFrame:
    """Читает Feather-файл в таблицу pandas"""
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()

    return table.to_pandas()


def _validate_frame(
    frame: pd.DataFrame,
    name: str,
    expected_columns: tuple[str, ...],
    id_column: str,
) -> None:
    """Проверяет базовую структуру входной таблицы"""
    if tuple(frame.columns) != expected_columns:
        raise ValueError(f"Некорректные колонки в таблице {name}")

    if frame.isna().to_numpy().any():
        raise ValueError(f"Таблица {name} содержит пропуски")

    if not frame[id_column].is_unique:
        raise ValueError(f"Колонка {name}.{id_column} содержит повторы")


def load_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает и проверяет статьи, calibration и test"""
    directory = Path(data_dir)
    articles = _read_feather(directory / ARTICLES_FILE)
    calibration = _read_feather(directory / CALIBRATION_FILE)
    test = _read_feather(directory / TEST_FILE)

    _validate_frame(articles, "articles", ARTICLE_COLUMNS, "article_id")
    _validate_frame(calibration, "calibration", CALIBRATION_COLUMNS, "query_id")
    _validate_frame(test, "test", TEST_COLUMNS, "query_id")

    known_article_ids = set(articles["article_id"])
    truth_article_ids = {
        article_id
        for value in calibration["ground_truth"]
        for article_id in parse_ground_truth(value)
    }
    unknown_article_ids = truth_article_ids - known_article_ids

    if unknown_article_ids:
        raise ValueError(
            f"В ground_truth найдены неизвестные статьи: {sorted(unknown_article_ids)}"
        )

    return articles, calibration, test


def prepare_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Добавляет очищенные заголовки и тексты статей"""
    prepared = articles.copy()
    prepared["clean_title"] = prepared["title"].map(normalize_text)
    prepared["clean_body"] = prepared["body"].map(html_to_text)

    if prepared["clean_body"].eq("").any():
        raise ValueError("После очистки найдены пустые статьи")

    return prepared


def split_text(
    value: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Разбивает текст на перекрывающиеся фрагменты"""
    if not 0 <= overlap < chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError("Некорректный размер фрагмента или перекрытия")

    words = value.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while len(words) - start > MAX_CHUNK_SIZE:
        chunks.append(" ".join(words[start : start + chunk_size]))
        start += step

    chunks.append(" ".join(words[start:]))

    return chunks


def build_article_chunks(
    articles: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> pd.DataFrame:
    """Формирует таблицу фрагментов с привязкой к статьям"""
    records: list[tuple[int, int, str]] = []
    rows = articles[["article_id", "clean_title", "clean_body"]].itertuples(
        index=False,
        name=None,
    )

    for article_id, title, body in rows:
        for chunk_id, chunk in enumerate(split_text(body, chunk_size, overlap)):
            records.append((int(article_id), chunk_id, f"{title}\n{chunk}"))

    chunks = pd.DataFrame.from_records(
        records,
        columns=("article_id", "chunk_id", "text"),
    )

    return chunks.astype({"article_id": "int64", "chunk_id": "int64", "text": "string"})
