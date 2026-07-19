import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config
from metrics import mean_average_precision_at_10
from retrieval import (
    build_label_matrix,
    cooccurrence_scores,
    dense_article_scores,
    dense_memory_scores,
    dense_query_similarity,
    fold_dense_memory_scores,
    fold_memory_scores,
    high_score_retrieval_scores,
    hybrid_retrieval_scores,
    leave_one_out_dense_memory_scores,
    leave_one_out_memory_scores,
    lexical_article_scores,
    lexical_query_similarity,
    load_data,
    logistic_oof_scores,
    logistic_query_scores,
    parse_ground_truth,
    prepare_articles,
    qwen_embedding_channels,
    qwen_reranker_scores,
    rank_article_ids,
)


def evaluate_calibration() -> None:
    """Проверяет retrieval на calibration выборке"""
    articles, calibration, _ = load_data()
    articles = prepare_articles(articles)
    article_ids = articles["article_id"].to_numpy()
    labels = build_label_matrix(calibration["ground_truth"], article_ids)
    queries = calibration["query_text"]
    ground_truth = calibration["ground_truth"].map(parse_ground_truth).tolist()
    similarity = lexical_query_similarity(queries)

    memory = leave_one_out_memory_scores(similarity, labels)
    embeddings = qwen_embedding_channels(articles, queries, queries)
    dense_direct = dense_article_scores(embeddings)
    dense_similarity = dense_query_similarity(embeddings)
    dense_memory = leave_one_out_dense_memory_scores(dense_similarity, labels)
    classifier = logistic_oof_scores(queries, labels, config.OOF_SEEDS[1])
    lexical_direct = lexical_article_scores(articles, queries)
    fusion = hybrid_retrieval_scores(
        lexical_direct,
        dense_direct,
        dense_memory,
        memory,
        classifier,
    )

    channels = {
        "lexical_memory": memory,
        "qwen_direct": dense_direct,
        "qwen_memory": dense_memory,
        "fusion": fusion,
    }
    reranker = qwen_reranker_scores(articles, queries, fusion)
    channels["high_score"] = high_score_retrieval_scores(fusion, reranker)

    for name, scores in channels.items():
        rankings = rank_article_ids(scores, article_ids)
        value = mean_average_precision_at_10(rankings, ground_truth)
        print(f"{name}: {value:.6f}")

    for seed in config.OOF_SEEDS:
        fold_memory = fold_memory_scores(similarity, labels, seed)
        dense_fold_memory = fold_dense_memory_scores(dense_similarity, labels, seed)
        classifier_fold = (
            classifier
            if seed == config.OOF_SEEDS[1]
            else logistic_oof_scores(queries, labels, seed)
        )
        fold_fusion = hybrid_retrieval_scores(
            lexical_direct,
            dense_direct,
            dense_fold_memory,
            fold_memory,
            classifier_fold,
        )
        rankings = rank_article_ids(fold_fusion, article_ids)
        value = mean_average_precision_at_10(rankings, ground_truth)
        print(f"fusion_seed_{seed}: {value:.6f}")


def build_submission(
    data_dir: str | Path,
    mode: str,
    cache_dir: str | Path,
    device: str,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Строит ранжированные ответы для test запросов"""
    articles, calibration, test = load_data(data_dir)
    articles = prepare_articles(articles)
    article_ids = articles["article_id"].to_numpy()
    labels = build_label_matrix(calibration["ground_truth"], article_ids)
    calibration_queries = calibration["query_text"]
    test_queries = test["query_text"]
    train_rows = np.arange(len(calibration))
    target_rows = np.arange(len(test))

    lexical_direct = lexical_article_scores(articles, test_queries)
    lexical_similarity = lexical_query_similarity(
        calibration_queries,
        test_queries,
    )
    lexical_memory = cooccurrence_scores(
        lexical_similarity,
        labels,
        train_rows,
        target_rows,
    )
    classifier = logistic_query_scores(
        calibration_queries,
        test_queries,
        labels,
        config.OOF_SEEDS[1],
    )
    embeddings = qwen_embedding_channels(
        articles,
        calibration_queries,
        test_queries,
        artifact_path=config.SUBMISSION_EMBEDDING_ARTIFACT,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
    )
    dense_direct = dense_article_scores(embeddings)
    dense_memory = dense_memory_scores(
        dense_query_similarity(embeddings),
        labels,
        train_rows,
        target_rows,
    )
    scores = hybrid_retrieval_scores(
        lexical_direct,
        dense_direct,
        dense_memory,
        lexical_memory,
        classifier,
    )

    if mode == config.HIGH_SCORE_MODE:
        reranker = qwen_reranker_scores(
            articles,
            test_queries,
            scores,
            artifact_path=config.SUBMISSION_RERANKER_ARTIFACT,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
        )
        scores = high_score_retrieval_scores(scores, reranker)

    rankings = rank_article_ids(scores, article_ids)
    submission = test[["query_id"]].copy()
    submission["answer"] = [" ".join(map(str, ranking)) for ranking in rankings]

    return submission, test, article_ids


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    article_ids: np.ndarray,
) -> None:
    """Проверяет контракт итогового CSV-файла"""
    if submission.columns.tolist() != ["query_id", "answer"]:
        raise ValueError("Ответ должен содержать колонки query_id и answer")

    if not submission["query_id"].equals(test["query_id"]):
        raise ValueError("Порядок query_id должен совпадать с test")

    known_ids = set(map(str, article_ids))

    for answer in submission["answer"]:
        values = answer.split()

        if len(values) != config.METRIC_CUTOFF or len(values) != len(set(values)):
            raise ValueError("Каждый ответ должен содержать 10 уникальных article_id")

        if not set(values) <= known_ids:
            raise ValueError("Ответ содержит неизвестный article_id")


def save_submission(submission: pd.DataFrame, output: str | Path) -> str:
    """Сохраняет CSV и возвращает его SHA-256"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, lineterminator="\n")

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments() -> argparse.Namespace:
    """Читает параметры командной строки"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=config.DEFAULT_OUTPUT_FILE)
    parser.add_argument("--cache-dir", type=Path, default=config.MODEL_CACHE_DIR)
    parser.add_argument("--mode", choices=config.RUN_MODES, default=config.FAST_MODE)
    parser.add_argument("--batch-size", type=int, default=config.EMBEDDING_BATCH_SIZE)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)

    return parser.parse_args()


def main() -> None:
    """Запускает построение и сохранение submission"""
    arguments = parse_arguments()
    submission, test, article_ids = build_submission(
        arguments.data_dir,
        arguments.mode,
        arguments.cache_dir,
        arguments.device,
        arguments.batch_size,
    )
    validate_submission(submission, test, article_ids)
    digest = save_submission(submission, arguments.output)
    print(f"Сохранено строк: {len(submission)}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
