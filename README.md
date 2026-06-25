# AI Coauthor

AI Coauthor — исследовательский прототип для курсовой работы: система строит карту научного
пространства по публикациям OpenAlex и рекомендует потенциальных соавторов с объяснениями на
основе LLM.

Проект связывает три части:

- semantic clustering статей: `UMAP -> HDBSCAN -> c-TF-IDF -> MMR -> LLM labeling`;
- author retrieval: mean embedding, Transformer, GraphSAGE по графу соавторства и гибрид
  Transformer+GraphSAGE;
- LLM reranking: из top-10 кандидатов выбираются 3 рекомендации, для них строятся объяснения с
  конкретными связями между статьями.

Основной текст работы находится в LaTeX: `main.tex`, `sections/*.tex`, `references.bib`.
Сборка пишет PDF в `build/main.pdf`. Вспомогательная документация лежит в `docs/`, итоговые
таблицы экспериментов — в `results/final_tables/`.

## Текущий результат

| Часть | Зафиксированный результат |
| --- | --- |
| Корпус | 509 447 публикаций OpenAlex по подполю Artificial Intelligence |
| Авторы для экспериментов | 35 954 квалифицированных автора |
| Тематическая карта | 979 fine-тем и 76 метакластеров |
| Retrieval budget | top-10 кандидатов перед LLM reranking |
| Финальная выдача | top-3 рекомендации после LLM reranking |
| LLM Has3@10, score >= 2 | 0.875 |
| LLM Selected3Mean@10 | 2.492 из 3 |
| Лучший observed nDCG@10 | 0.0102, только как proxy по будущим соавторам |

`observed nDCG@10` не используется как основное подтверждение качества рекомендаций: он считается
по будущим фактическим соавторствам и нужен как строгая, но разреженная proxy-оценка retrieval.
Качество финальной выдачи оценивается LLM-судьёй по релевантности кандидатов.

## Быстрый старт

Нужен Python 3.12+, `uv` и локальная установка LaTeX с `latexmk`, XeLaTeX и Biber.

```bash
uv sync
cp .env.example .env
```

Для LLM-разметки тем, LLM reranking, извлечения evidence и генерации финальных объяснений нужен
`OPENAI_API_KEY` в `.env`.

## Сборка текста работы

```bash
latexmk -xelatex main.tex
```

Сборка использует `.latexmkrc` и складывает временные файлы в `build/`. PDF не коммитится:
его можно воспроизвести из LaTeX-исходников.

## Данные и артефакты

Большие артефакты не входят в git: сырой OpenAlex-корпус, `*.npy` эмбеддинги, обученные модели,
LLM cache и LaTeX build. Репозиторий хранит код, конфиги, текст работы, figures для PDF и малые
таблицы результатов.

Ожидаемые локальные пути для полного запуска:

| Путь | Назначение |
| --- | --- |
| `data/openalex_clean.jsonl` | очищенные статьи |
| `data/embeddings.npy` | 768-мерные эмбеддинги статей |
| `data/paper_ids.json` | соответствие строк эмбеддингов и OpenAlex id |
| `data/clustering/` | результаты кластеризации статей и метакластеров |
| `data/authors/` | author splits и обучающие пары для retrieval |
| `models/` | чекпойнты Transformer/GraphSAGE |
| `results/` | метрики и итоговые таблицы |

Пустые директории для новых артефактов сохранены через `.gitkeep`.

## Основные команды

Команды ниже предполагают, что локальные данные уже лежат в ожидаемых путях.

```bash
# Эмбеддинги и тематическая карта
uv run python scripts/run_embed_papers.py
uv run python scripts/run_cluster_papers.py
uv run python scripts/run_meta_clustering.py --skip-llm

# Обучение и оценка author retrieval
uv run python scripts/run_train_recommender.py
uv run python scripts/run_eval_transformer_variants.py
uv run python scripts/run_graph_retrieval.py
uv run python scripts/run_hybrid_retrieval.py
uv run python scripts/run_retrieval_consistency.py

# LLM-оценка и финальные таблицы для PDF
uv run python scripts/run_eval_retrieval_llm.py
uv run python scripts/run_final_tables.py
```

## Веб-приложение

Backend написан на FastAPI, frontend — статический интерфейс на Deck.gl. Конфиг запуска:
`configs/app.yaml`.

```bash
uv run python scripts/run_eval_retrieval.py
uv run python scripts/run_measure_recs_coverage.py
uv run python scripts/run_app.py
```

По умолчанию приложение стартует на `http://127.0.0.1:8000`. Параметры `--host`, `--port` и
`--reload` переопределяют значения из конфига.

## Структура репозитория

| Путь | Содержание |
| --- | --- |
| `sections/` | главы курсовой в LaTeX |
| `figures/` | изображения, используемые в PDF |
| `docs/` | рабочая документация по архитектуре, методам, метрикам и экспериментам |
| `configs/` | YAML-конфиги кластеризации, рекомендаций и приложения |
| `scripts/` | thin entrypoints для запуска этапов пайплайна |
| `src/common/` | общие пути, I/O, YAML, кэш LLM, утилиты |
| `src/data/` | загрузка OpenAlex и подготовка author dataset |
| `src/embeddings/` | построение эмбеддингов статей и метакластеров |
| `src/clustering/` | кластеризация статей и построение метакластеров |
| `src/recommendation/` | Transformer, GraphSAGE, hybrid retrieval и LLM reranking |
| `src/evaluation/` | метрики кластеризации, retrieval и LLM-оценки |
| `src/llm/` | LLM-разметка тем и оценка рекомендаций |
| `src/web/` | FastAPI backend и статический frontend |
| `results/final_tables/` | итоговые таблицы, на которые опирается PDF |

## Где смотреть детали

- `docs/README.md` — навигация по рабочей документации.
- `docs/metrics_methods_pipeline.md` — выбранные метрики и согласованный evaluation pipeline.
- `docs/experiments.md` — экспериментальная глава в markdown-формате.
- `results/final_tables/tables.md` — сводные таблицы для PDF.
- `AGENTS.md` — правила редактирования текста курсовой.
