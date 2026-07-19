import config
from metrics import mean_average_precision_at_10
from retrieval import (
    build_label_matrix,
    dense_article_scores,
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
    memory_rankings = rank_article_ids(memory, article_ids)
    memory_map = mean_average_precision_at_10(memory_rankings, ground_truth)
    print(f"lexical_memory: {memory_map:.6f}")

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

    for name, scores in {
        "qwen_direct": dense_direct,
        "qwen_memory": dense_memory,
        "fusion": fusion,
    }.items():
        rankings = rank_article_ids(scores, article_ids)
        value = mean_average_precision_at_10(rankings, ground_truth)
        print(f"{name}: {value:.6f}")

    reranker = qwen_reranker_scores(articles, queries, fusion)
    high_score = high_score_retrieval_scores(fusion, reranker)
    high_score_rankings = rank_article_ids(high_score, article_ids)
    high_score_map = mean_average_precision_at_10(high_score_rankings, ground_truth)
    print(f"high_score: {high_score_map:.6f}")

    for seed in config.OOF_SEEDS:
        fold_memory = fold_memory_scores(similarity, labels, seed)
        dense_fold_memory = fold_dense_memory_scores(dense_similarity, labels, seed)
        logistic = (
            classifier
            if seed == config.OOF_SEEDS[1]
            else logistic_oof_scores(queries, labels, seed)
        )
        fold_fusion = hybrid_retrieval_scores(
            lexical_direct,
            dense_direct,
            dense_fold_memory,
            fold_memory,
            logistic,
        )

        memory_rankings = rank_article_ids(fold_memory, article_ids)
        logistic_rankings = rank_article_ids(logistic, article_ids)
        fusion_rankings = rank_article_ids(fold_fusion, article_ids)
        memory_map = mean_average_precision_at_10(memory_rankings, ground_truth)
        logistic_map = mean_average_precision_at_10(logistic_rankings, ground_truth)
        fusion_map = mean_average_precision_at_10(fusion_rankings, ground_truth)
        print(f"memory_seed_{seed}: {memory_map:.6f}")
        print(f"logistic_seed_{seed}: {logistic_map:.6f}")
        print(f"fusion_seed_{seed}: {fusion_map:.6f}")


if __name__ == "__main__":
    evaluate_calibration()
