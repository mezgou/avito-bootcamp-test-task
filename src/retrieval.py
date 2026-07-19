import unicodedata
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pyarrow as pa
from bs4 import BeautifulSoup
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

import config

type TextAnalyzer = Literal["word", "char_wb"]


def normalize_text(value: str) -> str:
    """Приводит текст к единому формату"""
    normalized = unicodedata.normalize("NFKC", value).translate(config.TEXT_TRANSLATION)
    normalized = "".join(
        character
        for character in normalized
        if character.isprintable() or character.isspace()
    )

    return config.WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_lexical_text(value: str) -> str:
    """Оставляет в тексте слова и числа для lexical поиска"""
    normalized = value.lower().translate(config.TEXT_TRANSLATION)

    return " ".join(config.LEXICAL_TOKEN_PATTERN.findall(normalized))


def html_to_text(value: str) -> str:
    """Извлекает полезный текст из HTML-разметки статьи"""
    soup = BeautifulSoup(value, "lxml")

    for element in soup.find_all(config.REMOVED_TAGS):
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
    data_dir: str | Path = config.DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает и проверяет статьи, calibration и test"""
    directory = Path(data_dir)
    articles = _read_feather(directory / config.ARTICLES_FILE)
    calibration = _read_feather(directory / config.CALIBRATION_FILE)
    test = _read_feather(directory / config.TEST_FILE)

    _validate_frame(articles, "articles", config.ARTICLE_COLUMNS, "article_id")
    _validate_frame(
        calibration,
        "calibration",
        config.CALIBRATION_COLUMNS,
        "query_id",
    )
    _validate_frame(test, "test", config.TEST_COLUMNS, "query_id")

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
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Разбивает текст на перекрывающиеся фрагменты"""
    if not 0 <= overlap < chunk_size <= config.MAX_CHUNK_SIZE:
        raise ValueError("Некорректный размер фрагмента или перекрытия")

    words = value.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while len(words) - start > config.MAX_CHUNK_SIZE:
        chunks.append(" ".join(words[start : start + chunk_size]))
        start += step

    chunks.append(" ".join(words[start:]))

    return chunks


def build_article_chunks(
    articles: pd.DataFrame,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
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


def tfidf_scores(
    corpus: pd.Series,
    queries: pd.Series,
    analyzer: TextAnalyzer,
    ngram_range: tuple[int, int],
    min_df: int,
    sublinear_tf: bool,
) -> np.ndarray:
    """Вычисляет TF-IDF сходство запросов и документов"""
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
        token_pattern=(
            config.WORD_TOKEN_PATTERN if analyzer == config.WORD_ANALYZER else None
        ),
    )
    corpus_matrix = vectorizer.fit_transform(corpus)
    query_matrix = vectorizer.transform(queries)

    return (query_matrix @ corpus_matrix.T).toarray()


def row_scale(scores: np.ndarray) -> np.ndarray:
    """Масштабирует оценки каждой строки относительно максимума"""
    maximum = np.maximum(
        scores.max(axis=1, keepdims=True),
        config.SCORE_EPSILON,
    )

    return scores / maximum


def lexical_score_channels(
    articles: pd.DataFrame,
    queries: pd.Series,
) -> dict[str, np.ndarray]:
    """Строит word и char оценки по текстам и заголовкам"""
    clean_queries = queries.map(normalize_text)

    body_word = tfidf_scores(
        articles["clean_body"],
        clean_queries,
        config.WORD_ANALYZER,
        config.WORD_NGRAM_RANGE,
        config.WORD_MIN_DF,
        config.WORD_SUBLINEAR_TF,
    )
    body_char = tfidf_scores(
        articles["clean_body"],
        clean_queries,
        config.CHAR_ANALYZER,
        config.CHAR_NGRAM_RANGE,
        config.BODY_CHAR_MIN_DF,
        config.CHAR_SUBLINEAR_TF,
    )
    title_word = tfidf_scores(
        articles["clean_title"],
        clean_queries,
        config.WORD_ANALYZER,
        config.WORD_NGRAM_RANGE,
        config.WORD_MIN_DF,
        config.WORD_SUBLINEAR_TF,
    )
    title_char = tfidf_scores(
        articles["clean_title"],
        clean_queries,
        config.CHAR_ANALYZER,
        config.CHAR_NGRAM_RANGE,
        config.TITLE_CHAR_MIN_DF,
        config.CHAR_SUBLINEAR_TF,
    )

    return {
        "body_word": body_word,
        "body_char": body_char,
        "title_word": title_word,
        "title_char": title_char,
    }


def lexical_article_scores(
    articles: pd.DataFrame,
    queries: pd.Series,
) -> np.ndarray:
    """Строит итоговые lexical оценки статей"""
    clean_queries = queries.map(normalize_text)
    scores = tfidf_scores(
        articles["clean_body"],
        clean_queries,
        config.WORD_ANALYZER,
        config.WORD_NGRAM_RANGE,
        config.WORD_MIN_DF,
        config.WORD_SUBLINEAR_TF,
    )

    return row_scale(scores)


def rank_article_ids(
    scores: np.ndarray,
    article_ids: np.ndarray,
    limit: int = config.METRIC_CUTOFF,
) -> np.ndarray:
    """Преобразует оценки в ранжированные article_id"""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :limit]

    return article_ids[order]


def build_label_matrix(
    ground_truth: pd.Series,
    article_ids: np.ndarray,
) -> np.ndarray:
    """Строит матрицу релевантности запросов и статей"""
    positions = {
        int(article_id): position for position, article_id in enumerate(article_ids)
    }
    labels = np.zeros((len(ground_truth), len(article_ids)), dtype=np.float32)

    for row, value in enumerate(ground_truth):
        for article_id in parse_ground_truth(value):
            labels[row, positions[article_id]] = 1.0

    return labels


def lexical_query_similarity(
    reference_queries: pd.Series,
    target_queries: pd.Series | None = None,
) -> np.ndarray:
    """Сравнивает запросы по словам и символьным фрагментам"""
    references = reference_queries.map(normalize_lexical_text)
    targets = (
        references
        if target_queries is None
        else target_queries.map(normalize_lexical_text)
    )

    word_vectorizer = TfidfVectorizer(
        ngram_range=config.WORD_NGRAM_RANGE,
        min_df=config.WORD_MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=config.WORD_TOKEN_PATTERN,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer=config.CHAR_ANALYZER,
        ngram_range=config.CHAR_NGRAM_RANGE,
        min_df=config.TITLE_CHAR_MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_references = word_vectorizer.fit_transform(references)
    char_references = char_vectorizer.fit_transform(references)
    word_scores = word_vectorizer.transform(targets) @ word_references.T
    char_scores = char_vectorizer.transform(targets) @ char_references.T
    similarity = (
        config.QUERY_WORD_WEIGHT * word_scores.toarray()
        + config.QUERY_CHAR_WEIGHT * char_scores.toarray()
    )

    if target_queries is None:
        np.fill_diagonal(similarity, 0.0)

    return similarity


def memory_scores(
    similarity: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    target_rows: np.ndarray,
    neighbors: int = config.MEMORY_NEIGHBORS,
    power: float = config.MEMORY_POWER,
    threshold: float = config.MEMORY_THRESHOLD,
    frequency_power: float = config.MEMORY_FREQUENCY_POWER,
) -> np.ndarray:
    """Переносит ответы от ближайших размеченных запросов"""
    current = similarity[np.ix_(target_rows, train_rows)]
    count = min(neighbors, len(train_rows))
    positions = np.argpartition(-current, count - 1, axis=1)[:, :count]
    neighbor_scores = np.take_along_axis(current, positions, axis=1)
    weights = np.maximum(neighbor_scores - threshold, 0.0) ** power
    neighbor_labels = labels[train_rows[positions]]
    scores = np.einsum("ij,ijk->ik", weights, neighbor_labels)
    frequencies = labels[train_rows].sum(axis=0)
    scores /= np.maximum(frequencies, 1.0) ** frequency_power

    return row_scale(scores)


def cooccurrence_scores(
    similarity: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    target_rows: np.ndarray,
) -> np.ndarray:
    """Расширяет перенесённые ответы совместно встречающимися статьями"""
    base = memory_scores(similarity, labels, train_rows, target_rows)
    frequencies = labels[train_rows].sum(axis=0)
    cooccurrence = labels[train_rows].T @ labels[train_rows]
    np.fill_diagonal(cooccurrence, 0.0)

    denominator = np.sqrt(np.outer(frequencies, frequencies))
    cosine = cooccurrence / np.maximum(denominator, config.SCORE_EPSILON)
    expansion = row_scale(base @ cosine)

    return row_scale(base + config.COOCCURRENCE_WEIGHT * expansion)


def leave_one_out_memory_scores(
    similarity: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Вычисляет query memory со строгим исключением текущего запроса"""
    scores = np.zeros_like(labels)
    all_rows = np.arange(len(labels))

    for row in all_rows:
        train_rows = np.delete(all_rows, row)
        scores[row] = cooccurrence_scores(
            similarity,
            labels,
            train_rows,
            np.array([row]),
        )[0]

    return scores


def fold_memory_scores(
    similarity: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Вычисляет query memory по разбиениям без пересечения"""
    scores = np.zeros_like(labels)
    splitter = KFold(
        n_splits=config.OOF_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    for train_rows, target_rows in splitter.split(labels):
        scores[target_rows] = cooccurrence_scores(
            similarity,
            labels,
            train_rows,
            target_rows,
        )

    return scores


def _query_features(
    train_queries: pd.Series,
    target_queries: pd.Series,
) -> tuple[csr_matrix, csr_matrix]:
    """Строит word и char признаки для классификатора"""
    word_vectorizer = TfidfVectorizer(
        ngram_range=config.WORD_NGRAM_RANGE,
        min_df=config.WORD_MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=config.WORD_TOKEN_PATTERN,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer=config.CHAR_ANALYZER,
        ngram_range=config.CHAR_NGRAM_RANGE,
        min_df=config.BODY_CHAR_MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_word = word_vectorizer.fit_transform(train_queries)
    train_char = char_vectorizer.fit_transform(train_queries)
    target_word = word_vectorizer.transform(target_queries)
    target_char = char_vectorizer.transform(target_queries)
    train_features = hstack(
        [train_word, config.CLASSIFIER_CHAR_WEIGHT * train_char],
        format="csr",
    )
    target_features = hstack(
        [target_word, config.CLASSIFIER_CHAR_WEIGHT * target_char],
        format="csr",
    )

    return train_features, target_features


def logistic_query_scores(
    train_queries: pd.Series,
    target_queries: pd.Series,
    train_labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Оценивает статьи независимыми логистическими классификаторами"""
    clean_train = train_queries.map(normalize_lexical_text)
    clean_target = target_queries.map(normalize_lexical_text)
    train_features, target_features = _query_features(clean_train, clean_target)
    seen = train_labels.sum(axis=0) > 0
    model = OneVsRestClassifier(
        LogisticRegression(
            C=config.CLASSIFIER_C,
            class_weight="balanced",
            solver="liblinear",
            max_iter=config.CLASSIFIER_MAX_ITERATIONS,
            random_state=seed,
        )
    )
    model.fit(train_features, train_labels[:, seen])
    probabilities = model.predict_proba(target_features)
    frequencies = train_labels[:, seen].sum(axis=0)
    prior = frequencies / max(float(frequencies.max()), 1.0)
    scores = np.zeros((len(target_queries), train_labels.shape[1]), dtype=np.float32)
    scores[:, seen] = row_scale(
        row_scale(probabilities) + config.CLASSIFIER_PRIOR_WEIGHT * prior
    )

    return scores


def logistic_oof_scores(
    queries: pd.Series,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Вычисляет OOF-оценки логистической классификации"""
    scores = np.zeros_like(labels)
    splitter = KFold(
        n_splits=config.OOF_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    for train_rows, target_rows in splitter.split(labels):
        scores[target_rows] = logistic_query_scores(
            queries.iloc[train_rows],
            queries.iloc[target_rows],
            labels[train_rows],
            seed,
        )

    return scores
