import gc
import hashlib
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from bs4 import BeautifulSoup
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import CrossEncoder, SentenceTransformer
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


def embedding_fingerprint(
    article_ids: Sequence[int],
    passages: Sequence[str],
    reference_queries: Sequence[str],
    target_queries: Sequence[str],
    device: str,
) -> str:
    """Вычисляет отпечаток входов и настроек embeddings"""
    digest = hashlib.sha256()
    settings = (
        str(config.EMBEDDING_CACHE_VERSION),
        config.EMBEDDING_MODEL,
        config.EMBEDDING_REVISION,
        str(config.EMBEDDING_MAX_LENGTH),
        str(config.EMBEDDING_PASSAGE_CHARACTERS),
        config.ARTICLE_QUERY_TASK,
        config.MEMORY_QUERY_TASK,
        device,
    )
    identifiers = tuple(str(int(article_id)) for article_id in article_ids)

    for values in (
        settings,
        identifiers,
        passages,
        reference_queries,
        target_queries,
    ):
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)

    return digest.hexdigest()


def article_passages(articles: pd.DataFrame) -> list[str]:
    """Формирует тексты статей для dense поиска"""
    rows = articles[["title", "body"]].itertuples(
        index=False,
        name=None,
    )

    return [
        f"{title}\n{html_to_dense_text(body)[: config.EMBEDDING_PASSAGE_CHARACTERS]}"
        for title, body in rows
    ]


def html_to_dense_text(value: str) -> str:
    """Извлекает непрерывный текст статьи для embeddings"""
    soup = BeautifulSoup(value, "lxml")

    for element in soup.find_all(config.REMOVED_TAGS):
        element.decompose()

    return soup.get_text(" ", strip=True)


def instructed_queries(queries: Sequence[str], task: str) -> list[str]:
    """Добавляет retrieval-инструкцию к запросам"""
    return [f"Instruct: {task}\nQuery: {query}" for query in queries]


def encode_embeddings(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Кодирует тексты в нормализованные embeddings"""
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32)


def _load_embedding_cache(
    path: Path,
    fingerprint: str,
    expected_rows: dict[str, int],
) -> dict[str, np.ndarray] | None:
    """Загружает embeddings при совпадении отпечатка"""
    if not path.exists():
        return None

    with np.load(path, allow_pickle=False) as stored:
        required = {
            "fingerprint",
            "passages",
            "reference_queries",
            "article_queries",
            "memory_queries",
        }
        if not required.issubset(stored.files):
            return None

        if str(stored["fingerprint"].item()) != fingerprint:
            return None

        channels = {name: stored[name] for name in required if name != "fingerprint"}

    widths = {channel.shape[1] for channel in channels.values() if channel.ndim == 2}
    valid = (
        widths == {config.EMBEDDING_DIMENSION}
        and all(
            channel.ndim == 2
            and channel.shape[0] == expected_rows[name]
            and np.isfinite(channel).all()
            for name, channel in channels.items()
        )
        and all(
            np.allclose(np.linalg.norm(channel, axis=1), 1.0, atol=1e-3)
            for channel in channels.values()
        )
    )

    return channels if valid else None


def qwen_embedding_channels(
    articles: pd.DataFrame,
    reference_queries: pd.Series,
    target_queries: pd.Series,
    artifact_path: str | Path = config.EMBEDDING_ARTIFACT,
    cache_dir: str | Path = config.MODEL_CACHE_DIR,
    device: str = config.DEFAULT_DEVICE,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Строит и кэширует Qwen embeddings для retrieval"""
    passages = article_passages(articles)
    references = reference_queries.tolist()
    targets = target_queries.tolist()
    actual_device = (
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    fingerprint = embedding_fingerprint(
        articles["article_id"].tolist(),
        passages,
        references,
        targets,
        actual_device,
    )
    artifact = Path(artifact_path)
    expected_rows = {
        "passages": len(passages),
        "reference_queries": len(references),
        "article_queries": len(targets),
        "memory_queries": len(targets),
    }
    cached = _load_embedding_cache(artifact, fingerprint, expected_rows)

    if cached is not None:
        return cached

    model_kwargs: dict[str, object] = {"attn_implementation": "sdpa"}
    if actual_device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16

    model = SentenceTransformer(
        config.EMBEDDING_MODEL,
        revision=config.EMBEDDING_REVISION,
        device=actual_device,
        cache_folder=str(cache_dir),
        model_kwargs=model_kwargs,
        processor_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = config.EMBEDDING_MAX_LENGTH
    channels = {
        "passages": encode_embeddings(model, passages, batch_size),
        "reference_queries": encode_embeddings(model, references, batch_size),
        "article_queries": encode_embeddings(
            model,
            instructed_queries(targets, config.ARTICLE_QUERY_TASK),
            batch_size,
        ),
        "memory_queries": encode_embeddings(
            model,
            instructed_queries(targets, config.MEMORY_QUERY_TASK),
            batch_size,
        ),
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_name(f"{artifact.stem}.tmp.npz")
    np.savez_compressed(temporary, fingerprint=np.array(fingerprint), **channels)
    temporary.replace(artifact)

    del model
    gc.collect()
    if actual_device.startswith("cuda"):
        torch.cuda.empty_cache()

    return channels


def dense_article_scores(channels: dict[str, np.ndarray]) -> np.ndarray:
    """Вычисляет dense оценки статей"""
    return row_scale(channels["article_queries"] @ channels["passages"].T)


def dense_query_similarity(channels: dict[str, np.ndarray]) -> np.ndarray:
    """Вычисляет сходство с размеченными запросами"""
    return channels["memory_queries"] @ channels["reference_queries"].T


def dense_memory_scores(
    similarity: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    target_rows: np.ndarray,
) -> np.ndarray:
    """Переносит ответы по сходству Qwen embeddings"""
    return memory_scores(
        similarity,
        labels,
        train_rows,
        target_rows,
        power=config.DENSE_MEMORY_POWER,
        threshold=config.DENSE_MEMORY_THRESHOLD,
        frequency_power=config.DENSE_MEMORY_FREQUENCY_POWER,
    )


def leave_one_out_dense_memory_scores(
    similarity: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Вычисляет Qwen query memory со строгим LOO"""
    scores = np.zeros_like(labels)
    all_rows = np.arange(len(labels))

    for row in all_rows:
        train_rows = np.delete(all_rows, row)
        scores[row] = dense_memory_scores(
            similarity,
            labels,
            train_rows,
            np.array([row]),
        )[0]

    return scores


def fold_dense_memory_scores(
    similarity: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Вычисляет Qwen query memory по OOF-разбиениям"""
    scores = np.zeros_like(labels)
    splitter = KFold(
        n_splits=config.OOF_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    for train_rows, target_rows in splitter.split(labels):
        scores[target_rows] = dense_memory_scores(
            similarity,
            labels,
            train_rows,
            target_rows,
        )

    return scores


def hybrid_retrieval_scores(
    lexical_direct: np.ndarray,
    dense_direct: np.ndarray,
    dense_memory: np.ndarray,
    lexical_memory: np.ndarray,
    classifier: np.ndarray,
) -> np.ndarray:
    """Объединяет lexical, dense и calibration оценки"""
    direct = row_scale(
        config.DIRECT_LEXICAL_WEIGHT * lexical_direct
        + config.DIRECT_DENSE_WEIGHT * dense_direct
    )
    scores = (
        config.FUSION_DIRECT_WEIGHT * direct
        + config.FUSION_DENSE_MEMORY_WEIGHT * dense_memory
        + config.FUSION_LEXICAL_MEMORY_WEIGHT * lexical_memory
        + config.FUSION_CLASSIFIER_WEIGHT * classifier
    )

    return row_scale(scores)


def reranker_candidate_indices(
    scores: np.ndarray,
    count: int = config.RERANKER_CANDIDATES,
) -> np.ndarray:
    """Выбирает упорядоченные позиции кандидатов для reranker"""
    candidates = np.argpartition(-scores, count - 1, axis=1)[:, :count]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)

    return np.take_along_axis(candidates, order, axis=1)


def reranker_chunks(articles: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Разбивает статьи на фрагменты для reranker"""
    texts: list[str] = []
    owners: list[int] = []
    step = config.RERANKER_CHUNK_SIZE - config.RERANKER_CHUNK_OVERLAP
    rows = articles[["title", "body"]].itertuples(index=False, name=None)

    for owner, (title, body) in enumerate(rows):
        words = html_to_dense_text(body).split()
        starts = range(0, max(len(words), 1), step)

        for start in starts:
            chunk = " ".join(words[start : start + config.RERANKER_CHUNK_SIZE])
            texts.append(f"{title}\n{chunk}")
            owners.append(owner)

    return texts, np.asarray(owners, dtype=np.int64)


def select_candidate_passages(
    articles: pd.DataFrame,
    queries: pd.Series,
    candidates: np.ndarray,
) -> list[str]:
    """Выбирает наиболее подходящий фрагмент каждого кандидата"""
    chunks, owners = reranker_chunks(articles)
    clean_chunks = pd.Series(chunks).map(normalize_lexical_text)
    clean_queries = queries.map(normalize_lexical_text)
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
        max_features=config.RERANKER_CHAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_chunks = word_vectorizer.fit_transform(clean_chunks)
    char_chunks = char_vectorizer.fit_transform(clean_chunks)
    word_scores = (word_vectorizer.transform(clean_queries) @ word_chunks.T).toarray()
    char_scores = (char_vectorizer.transform(clean_queries) @ char_chunks.T).toarray()
    selection_scores = (
        config.CHUNK_WORD_WEIGHT * word_scores + config.CHUNK_CHAR_WEIGHT * char_scores
    )
    owner_chunks = {
        int(owner): np.flatnonzero(owners == owner) for owner in np.unique(owners)
    }
    selected: list[str] = []

    for query_row, row_candidates in enumerate(candidates):
        for candidate in row_candidates:
            indexes = owner_chunks[int(candidate)]
            best = indexes[np.argmax(selection_scores[query_row, indexes])]
            selected.append(chunks[int(best)])

    return selected


def reranker_fingerprint(
    articles: pd.DataFrame,
    queries: pd.Series,
    device: str,
) -> str:
    """Вычисляет отпечаток данных и настроек reranker"""
    digest = hashlib.sha256()
    settings = (
        str(config.RERANKER_CACHE_VERSION),
        config.RERANKER_MODEL,
        config.RERANKER_REVISION,
        str(config.RERANKER_MAX_LENGTH),
        str(config.RERANKER_CHUNK_SIZE),
        str(config.RERANKER_CHUNK_OVERLAP),
        str(config.WORD_NGRAM_RANGE),
        str(config.CHAR_NGRAM_RANGE),
        str(config.WORD_MIN_DF),
        str(config.BODY_CHAR_MIN_DF),
        str(config.RERANKER_CHAR_MAX_FEATURES),
        str(config.CHUNK_WORD_WEIGHT),
        str(config.CHUNK_CHAR_WEIGHT),
        config.LEXICAL_TOKEN_PATTERN.pattern,
        config.RERANKER_TASK,
        device,
    )
    article_values = (
        f"{article_id}\n{title}\n{body}"
        for article_id, title, body in articles[
            ["article_id", "title", "body"]
        ].itertuples(index=False, name=None)
    )

    for values in (settings, article_values, queries.tolist()):
        for value in values:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)

    return digest.hexdigest()


def _load_reusable_reranker_logits(
    path: Path,
    fingerprint: str,
    candidates: np.ndarray,
) -> np.ndarray:
    """Переносит закэшированные logits общих кандидатов"""
    logits = np.full(candidates.shape, np.nan, dtype=np.float32)
    if not path.exists():
        return logits

    with np.load(path, allow_pickle=False) as stored:
        required = {"fingerprint", "candidates", "logits"}
        if not required.issubset(stored.files):
            return logits

        cached_candidates = stored["candidates"]
        cached_logits = stored["logits"]
        valid = (
            str(stored["fingerprint"].item()) == fingerprint
            and cached_candidates.shape == cached_logits.shape
            and cached_candidates.shape[0] == candidates.shape[0]
            and np.isfinite(cached_logits).all()
        )
        if not valid:
            return logits

    for row, row_candidates in enumerate(candidates):
        cached = {
            int(candidate): float(logit)
            for candidate, logit in zip(
                cached_candidates[row],
                cached_logits[row],
                strict=True,
            )
        }

        for position, candidate in enumerate(row_candidates):
            if int(candidate) in cached:
                logits[row, position] = cached[int(candidate)]

    return logits


def reranker_minmax_scores(
    candidates: np.ndarray,
    logits: np.ndarray,
    width: int,
) -> np.ndarray:
    """Масштабирует logits reranker внутри списка кандидатов"""
    minimum = logits.min(axis=1, keepdims=True)
    scale = np.maximum(
        logits.max(axis=1, keepdims=True) - minimum,
        config.SCORE_EPSILON,
    )
    normalized = (logits - minimum) / scale
    scores = np.zeros((len(candidates), width), dtype=np.float32)

    for row, row_candidates in enumerate(candidates):
        scores[row, row_candidates] = normalized[row]

    return scores


def qwen_reranker_scores(
    articles: pd.DataFrame,
    queries: pd.Series,
    base_scores: np.ndarray,
    artifact_path: str | Path = config.RERANKER_ARTIFACT,
    cache_dir: str | Path = config.MODEL_CACHE_DIR,
    device: str = config.DEFAULT_DEVICE,
    batch_size: int = config.RERANKER_BATCH_SIZE,
) -> np.ndarray:
    """Оценивает top-кандидатов Qwen reranker и кэширует logits"""
    actual_device = (
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    candidates = reranker_candidate_indices(base_scores)
    fingerprint = reranker_fingerprint(articles, queries, actual_device)
    artifact = Path(artifact_path)
    logits = _load_reusable_reranker_logits(artifact, fingerprint, candidates)
    missing = np.flatnonzero(~np.isfinite(logits.ravel()))

    if len(missing):
        passages = select_candidate_passages(articles, queries, candidates)
        repeated_queries = [
            str(query) for query in queries for _ in range(config.RERANKER_CANDIDATES)
        ]
        pairs = [(repeated_queries[index], passages[index]) for index in missing]
        model_kwargs: dict[str, object] = {"attn_implementation": "sdpa"}
        if actual_device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16

        model = CrossEncoder(
            config.RERANKER_MODEL,
            revision=config.RERANKER_REVISION,
            device=actual_device,
            cache_folder=str(cache_dir),
            max_length=config.RERANKER_MAX_LENGTH,
            model_kwargs=model_kwargs,
            processor_kwargs={"padding_side": "left"},
        )
        predictions = model.predict(
            pairs,
            prompt=config.RERANKER_TASK,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        logits.ravel()[missing] = np.asarray(predictions, dtype=np.float32)

        del model
        gc.collect()
        if actual_device.startswith("cuda"):
            torch.cuda.empty_cache()

        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_name(f"{artifact.stem}.tmp.npz")
        np.savez_compressed(
            temporary,
            fingerprint=np.array(fingerprint),
            candidates=candidates,
            logits=logits,
        )
        temporary.replace(artifact)

    return reranker_minmax_scores(candidates, logits, base_scores.shape[1])


def high_score_retrieval_scores(
    base_scores: np.ndarray,
    reranker_scores: np.ndarray,
) -> np.ndarray:
    """Добавляет оценки reranker к базовому ранжированию"""
    return (
        config.HIGH_SCORE_BASE_WEIGHT * base_scores
        + config.HIGH_SCORE_RERANKER_WEIGHT * reranker_scores
    )
