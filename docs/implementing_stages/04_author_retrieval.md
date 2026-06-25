# Глава 2.5. Рекомендательная система: retrieval совместимых исследователей

## Корректная постановка задачи

Подсистема рекомендаций решает задачу **retrieval семантически совместимых исследователей**:
по истории публикаций автора найти других исследователей с близкими или взаимодополняемыми научными
интересами.

Важно **не** формулировать это как «предсказание будущих соавторов»: система не прогнозирует
конкретное событие соавторства, а **извлекает релевантных кандидатов** из пространства авторов.
Реальные соавторства покрывают лишь часть всех релевантных связей (см. главу 3.4, проблема
разреженности), поэтому формулировка через retrieval честнее и устойчивее.

## Общий pipeline

```
author history
  → mean / Transformer / GraphSAGE author embedding
  → retrieval (поиск кандидатов по близости)
  → LLM reranking (переранжирование)
  → explanation generation (объяснение)
```

```mermaid
flowchart LR
    hist[История статей автора] --> model[Transformer encoder]
    hist --> graph[GraphSAGE over coauthor graph]
    model --> aemb[Эмбеддинг автора]
    graph --> aemb
    aemb --> retr[Dense retrieval: топ-N по косинусу]
    retr --> rerank[LLM reranking]
    rerank --> expl[Объяснения]
```

В этой главе описаны первые два шага (представление автора и retrieval); LLM-переранжирование и
генерация объяснений — в [главе 2.6](05_llm_reranking.md).

## Представление автора: mean, Transformer, GraphSAGE

В работе сравниваются три источника сигнала.

- `mean_author_embedding` — L2-нормализованное среднее последних статей автора до cutoff. Это
  простой semantic baseline без обучения.
- Transformer Encoder — агрегирует последовательность эмбеддингов статей автора, упорядоченных по
  времени.
- GraphSAGE — стартует из mean-признаков авторов и делает message passing по графу прошлых
  соавторств. В метакластерном варианте к графу добавляются 76 meta-узлов.

Ключевые элементы архитектуры (`recommendation/model.py`):

- **Проекция входа.** Каждая статья истории подаётся как конкатенация её эмбеддинга и дополнительных
  признаков, проецируемая в скрытое пространство `d_model` (финальная модель — 256).
- **Временные дельты.** Разница во времени между статьями кодируется через обучаемые эмбеддинги
  «бакетов дельт» (`delta_emb`), чтобы модель учитывала динамику интересов автора.
- **Author token.** К последовательности добавляется специальный обучаемый токен (по аналогии с
  `[CLS]`), чьё выходное представление и служит эмбеддингом автора.
- **Кодировщик.** Стек слоёв `nn.TransformerEncoder`.
- **Residual-from-mean голова.** Голова проецирует представление автора в пространство эмбеддингов
  статей (`emb_dim`), но как **остаток поверх среднего истории**:
  `pred = normalize(mean(history) + res_scale · head(h_author))`. Параметр `res_scale` стартует с
  нуля → на инициализации модель воспроизводит mean-бейзлайн и затем учится отклоняться от него.
  Это позволяет Transformer выделять кластерную структуру интересов автора (а не «усреднять» их в
  одну точку, как простой mean), отталкиваясь от сильного семантического приближения.

## Датасет авторов

1. Квалифицированные авторы (≥10 статей и ≥10 статей в истории до cutoff 2024) → `author_index.json`.
2. Split **80/10/10** по author_id → `author_splits.json`.
3. Для каждого автора в сплите: история `year ≤ 2024`, таргет `year > 2024` → `*_examples.jsonl`.

```bash
uv run python -m data.build_author_dataset --force
```

## Обучение

Единый loss — **multi-positive coauthor-InfoNCE** (`recommendation/coauthor_loss.py`):
focal author (история → Transformer) против будущих новых соавторов (positives) и shared in-batch
random negatives, всё кодируется тем же Transformer **без detach** (симметрия + негативы не дают
коллапса). Модель: `author_retriever`.

Ключевой приём — **residual-from-mean** (`recommendation/model.py`):
`pred = normalize(mean(history) + res_scale · head(h_author))`, `res_scale` инициализируется нулём, так
что на старте обучаемый вектор **равен** mean-бейзлайну, а обучение лишь уточняет его поверх богатых
pretrained-эмбеддингов статей.

- **Расписание:** warmup + cosine LR для Transformer; early stopping по val nDCG@50.
- **Артефакты:** `models/<model_name>/best.pt`, `train_log.json`.

Финальные Transformer-варианты используют `d_model=256`, 2 слоя, 4 heads, dropout 0.2,
`n_negatives=256`, `tau=0.05`, `max_history=20`. GraphSAGE использует 2 слоя, hidden_dim 256,
dropout 0.2 и тот же `tau=0.05`.

```bash
uv run python -m recommendation.train --force \
  --d-model 256 --n-layers 2 --n-heads 4 --dim-feedforward 512 \
  --dropout 0.2 --weight-decay 0.05 --lr 2e-4 --tau 0.05 --n-negatives 256
```

После обучения `predict_author_embeddings.py` сохраняет `author_embeddings.npy` и `author_ids.json`
в директории соответствующей модели.

## Stage 1: Dense retrieval

Первый этап рекомендаций — быстрый отбор кандидатов по близости в пространстве предсказанных
эмбеддингов авторов (`recommendation/pipeline.py`):

1. Предсказанные эмбеддинги авторов L2-нормализуются.
2. Для запрашиваемого автора вычисляется косинусная близость со всеми остальными авторами
   (скалярное произведение нормализованных векторов).
3. Исключаются сам автор и его уже существующие соавторы — рекомендуем **новые** связи.
4. Берутся топ-10 кандидатов — это вход для LLM-переранжирования, которое выбирает 3 автора.

## Оценка retrieval и бейзлайны

Скрипт `evaluation/eval_retrieval.py` сравнивает обучаемую модель с бейзлайнами на **author-level
retrieval** (`evaluation/coauthor_retrieval.py`) — в согласовании с prod-pipeline:

- **Кандидаты:** авторы **того же сплита** (train/val/test), кроме запрашивающего и уже известных соавторов.
- **Релевантные:** **новые** coauthors на статьях **после 2024**, из пула сплита.
- **Скоринг:** косинус query-вектора (Transformer или mean истории) с вектором кандидата.

Бейзлайны:

- **`random`** — случайное ранжирование кандидатов-авторов;
- **`mean_author_embedding`** — L2-нормализованное среднее эмбеддингов статей истории (для query и
  каждого кандидата).

Результаты: `results/retrieval/consistency/retrieval_metrics_at_k.csv`,
`results/retrieval/transformer_variants_metrics.csv`,
`results/retrieval/consistency/llm_threshold_metrics_at_k.csv`. См. главу 3.3 для чисел.

## Кто считается «квалифицированным» автором

Рекомендации строятся для авторов с достаточной историей публикаций (≥10 статей и ≥10 статей в
истории до cutoff 2024). Всего квалифицированных авторов — **35 954**. Для них собраны истории,
mean-, Transformer- и GraphSAGE-представления.

## Результат этапа

- `models/author_retriever/best.pt` — веса модели автора.
- `models/transformer_author_no_cluster/best.pt` — Transformer без кластерных признаков.
- `models/graphsage_author/best.pt` — GraphSAGE по графу соавторства.
- `models/graphsage_author_metacluster/best.pt` — GraphSAGE с 76 meta-узлами.
- `models/graphsage_transformer_author/best.pt` — hybrid Transformer+GraphSAGE.

## Иллюстрации для этой главы

- Кривые обучения модели: `models/author_retriever/train_loss.png`, `val_loss.png`,
  `val_ndcg10.png`.
- Схема архитектуры Transformer-представления автора (можно отрисовать по описанию выше).
- Диаграмма pipeline рекомендаций (mermaid выше).
