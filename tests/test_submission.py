import numpy as np
import pandas as pd
import pytest

from solution import validate_submission


def valid_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Создаёт корректные синтетические таблицы"""
    test = pd.DataFrame({"query_id": [2, 1], "query_text": ["a", "b"]})
    answer = " ".join(map(str, range(1, 11)))
    submission = pd.DataFrame({"query_id": [2, 1], "answer": [answer, answer]})
    article_ids = np.arange(1, 11)

    return submission, test, article_ids


def test_validate_submission_accepts_contract() -> None:
    """Проверяет корректный формат submission"""
    submission, test, article_ids = valid_frames()

    validate_submission(submission, test, article_ids)


def test_validate_submission_rejects_query_order() -> None:
    """Проверяет порядок идентификаторов запросов"""
    submission, test, article_ids = valid_frames()
    submission["query_id"] = submission["query_id"].iloc[::-1].to_numpy()

    with pytest.raises(ValueError, match="Порядок query_id"):
        validate_submission(submission, test, article_ids)


def test_validate_submission_rejects_duplicate_articles() -> None:
    """Проверяет уникальность статей внутри ответа"""
    submission, test, article_ids = valid_frames()
    submission.loc[0, "answer"] = "1 1 2 3 4 5 6 7 8 9"

    with pytest.raises(ValueError, match="10 уникальных"):
        validate_submission(submission, test, article_ids)
