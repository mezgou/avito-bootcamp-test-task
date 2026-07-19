import config
from metrics import mean_average_precision_at_10
from retrieval import (
    build_label_matrix,
    fold_memory_scores,
    leave_one_out_memory_scores,
    lexical_query_similarity,
    load_data,
    logistic_oof_scores,
    parse_ground_truth,
    rank_article_ids,
)


def evaluate_calibration() -> None:
    """Проверяет query memory на calibration выборке"""
    articles, calibration, _ = load_data()
    article_ids = articles["article_id"].to_numpy()
    labels = build_label_matrix(calibration["ground_truth"], article_ids)
    queries = calibration["query_text"]
    ground_truth = calibration["ground_truth"].map(parse_ground_truth).tolist()
    similarity = lexical_query_similarity(queries)

    memory = leave_one_out_memory_scores(similarity, labels)
    memory_rankings = rank_article_ids(memory, article_ids)
    memory_map = mean_average_precision_at_10(memory_rankings, ground_truth)
    print(f"lexical_memory: {memory_map:.6f}")

    for seed in config.OOF_SEEDS:
        fold_memory = fold_memory_scores(similarity, labels, seed)
        logistic = logistic_oof_scores(queries, labels, seed)

        memory_rankings = rank_article_ids(fold_memory, article_ids)
        logistic_rankings = rank_article_ids(logistic, article_ids)
        memory_map = mean_average_precision_at_10(memory_rankings, ground_truth)
        logistic_map = mean_average_precision_at_10(logistic_rankings, ground_truth)
        print(f"memory_seed_{seed}: {memory_map:.6f}")
        print(f"logistic_seed_{seed}: {logistic_map:.6f}")


if __name__ == "__main__":
    evaluate_calibration()
