# avito-bootcamp-test-task
Avito Data Science Bootcamp - Тестовое задание

## Задача

Для каждого вопроса пользователя необходимо вернуть до 10 релевантных статей
справочного центра в порядке убывания полезности. Целевая метрика — MAP@10

## Решение

Решение полностью работает локально и объединяет несколько методов поиска:

1. удаление `script`, `style`, `input` и `noscript` из HTML с сохранением заголовков,
   таблиц, вкладок, спойлеров и `img alt`
2. прямой поиск по статьям через word TF-IDF
3. перенос разметки от похожих calibration-запросов
4. One-vs-Rest Logistic Regression по word и char TF-IDF
5. dense retrieval через `Qwen/Qwen3-Embedding-0.6B`
6. адаптивное снижение веса query memory для новых и непохожих запросов
7. переранжирование top-15 через `Qwen/Qwen3-Reranker-0.6B`

## Валидация

Query memory проверяется со строгим leave-one-out. Для более честной оценки
дополнительно используется repeated 5-fold OOF по трём seed. Test-разметка,
ручная корректировка ответов и привязка к `query_id` не используются

| Вариант | MAP@10 |
|---|---:|
| Word TF-IDF | 0.324200 |
| Qwen query memory | 0.654318 |
| Fusion, repeated OOF | 0.661255–0.670616 |
| Robust fusion, repeated OOF | 0.680608–0.684955 |
| Fusion + reranker, LOO | 0.695564 |
| Public LB, robust + reranker | 0.630129 |

LOO-результат оптимистичен, поэтому для дальнейшей настройки используются
repeated OOF и проверки устойчивости к статьям, отсутствующим в train-fold

## Анализ ошибок

- служебные HTML-элементы создавали шум, поэтому они удаляются до индексации
- query memory ошибалась на новых темах, поэтому её вес зависит от сходства запросов
- релевантные статьи терялись из-за порядка кандидатов, поэтому top-15
  дополнительно обрабатывается reranker

## Запуск

Установка окружения:

```bash
uv sync --locked
```

Быстрый режим:

```bash
uv run python src/solution.py --mode fast --output answer.csv
```

Режим с reranker:

```bash
uv run python src/solution.py --mode high-score --output answer.csv
```

Рекомендуемый устойчивый режим:

```bash
uv run python src/solution.py --mode robust --output answer.csv
```

Доступные параметры: `--data-dir`, `--output`, `--cache-dir`, `--mode`,
`--batch-size` и `--device`

## Проверки

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Ноутбук [notebooks/solution.ipynb](notebooks/solution.ipynb) содержит EDA,
абляции, candidate recall, срезы ошибок и воспроизводит итоговый submission

## Конфигурация

Решение запускалось с NVIDIA RTX 3060 6 GB Laptop, AMD Ryzen 5 5600H
и 16 GB RAM
