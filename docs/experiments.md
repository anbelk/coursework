# Глава 3. Эксперименты

Все числа в этой главе взяты из реальных артефактов запусков (`data/`, `results/`). Экспериментальный
контур — корпус подполя **Artificial Intelligence** в OpenAlex (509k статей).

## 3.1. Датасет

Источник — OpenAlex, фильтр по подполю AI. Для retrieval отбираются **квалифицированные авторы**
(≥ 10 публикаций, ≥ 10 статей в истории до cutoff 2024), затем сохраняются только их статьи.

| Характеристика | Значение |
| --- | --- |
| Статей после фильтра | 509 447 |
| Квалифицированных авторов (≥ 10 статей) | 35 954 |
| Размерность эмбеддинга | 768 (`intfloat/multilingual-e5-base`) |
| Префикс эмбеддера | `query: ` + title + abstract |
| UMAP для кластеризации | `n_components = 10` |
| Soft membership автора `q` | 980 (fine-кластеры + noise) |

**Разбиение для retrieval** — двухэтапное (`data/authors/dataset_meta.json`):

1. **По авторам:** 80% / 10% / 10% → train / val / test (seed = 42).
2. **По годам внутри каждого автора:** история — публикации **≤ 2024**, таргет — **> 2024**.
   Таргетные **соавторы** — новые квалифицированные coauthors из будущих статей (не из прошлой
   истории).

| Split | Примеров | Примеров с ≥1 релевантным coauthor |
| --- | --- | --- |
| train | 16 210 | 7 773 (для обучения) |
| val | 2 021 | 280 |
| test | 1 973 | **279** |

Ранний прогон с порогом ≥ 5 статей давал больше релевантных test-примеров, но короткие истории
(5–10 статей) делали mean baseline слишком сильным; фильтр ≥ 10 улучшил относительное качество
Transformer.

---

## 3.2. Кластеризация: HDBSCAN, FCM, GMM и бейзлайны

### Pipeline

```
embeddings → UMAP-10 → {HDBSCAN | FCM | GMM | random | k-means}
          → c-TF-IDF → MMR → LLM labels → coherence / distinctness / quant
```

Сравнивались три семейства алгоритмов и сетка наивных бейзлайнов **`random_umap10_k*`** /
**`kmeans_umap10_k*`** на UMAP-10 с K ∈ {50, 100, 200, 400, 600, 800, 1000}.

### Количественные метрики на AI-корпусе (509k)

Источник: `data/topics/*/metrics_quant.json`, сводка `results/summary_metrics_quant.csv`.

| variant | K | entropy_norm | size_p50 | top5_conc | noise |
| --- | --- | --- | --- | --- | --- |
| **hdbscan_fine** | **979** | 0.019 | 112 | 0.032 | 0.647 |
| hdbscan_medium | 242 | 0.032 | 580 | 0.253 | 0.358 |
| hdbscan_coarse | 132 | 0.037 | 1235 | 0.262 | 0.312 |
| gmm_200 | 200 | 0.020 | 203 | 0.063 | 0.000 |
| gmm_100 | 100 | 0.032 | 395 | 0.134 | 0.000 |
| gmm_50 | 50 | 0.049 | 713 | 0.253 | 0.000 |
| fcm_200 | 183 | 1.000 | 8 | 0.882 | 0.000 |
| fcm_100 | 100 | 1.000 | 87 | 0.649 | 0.000 |
| fcm_50 | 50 | 1.000 | 188 | 0.708 | 0.000 |
| kmeans_umap10_k1000 | 1000 | 0.000 | 428 | 0.023 | 0.000 |
| kmeans_umap10_k800 | 800 | 0.000 | 540 | 0.025 | 0.000 |
| kmeans_umap10_k400 | 400 | 0.000 | 1092 | 0.046 | 0.000 |
| kmeans_umap10_k200 | 200 | 0.000 | 2351 | 0.075 | 0.000 |
| kmeans_umap10_k100 | 100 | 0.000 | 4984 | 0.122 | 0.000 |
| kmeans_umap10_k50 | 50 | 0.000 | 10375 | 0.202 | 0.000 |
| random_umap10_k1000 | 1000 | 0.000 | 509 | 0.006 | 0.000 |
| random_umap10_k800 | 800 | 0.000 | 636 | 0.007 | 0.000 |
| random_umap10_k400 | 400 | 0.000 | 1273 | 0.013 | 0.000 |
| random_umap10_k200 | 200 | 0.000 | 2546 | 0.026 | 0.000 |
| random_umap10_k100 | 100 | 0.000 | 5090 | 0.051 | 0.000 |
| random_umap10_k50 | 50 | 0.000 | 10187 | 0.102 | 0.000 |

**Наблюдения на масштабе 509k:**

- HDBSCAN даёт много мелких тем (K = 979) ценой высокой доли шума (0.647): значительная часть
  статей не попадает в плотные кластеры — ожидаемо для неоднородного AI-корпуса.
- FCM даёт максимальную энтропию назначений (1.0) и высокую концентрацию в top-5 кластерах
  (до 0.88) — типичный признак fail-guards на этом корпусе.
- K-Means / random с фиксированным K дают полное покрытие (noise = 0) и нулевую энтропию
  (hard assignment); кривая по K видна в `top5_conc` и `size_p50`.
- Для prod выбран **`hdbscan_fine`** — баланс числа тем и детализации; membership через
  `probabilities_` (без materialize N×K матрицы).

### LLM-метрики на AI-корпусе (509k)

**Quant + c-TF-IDF/MMR** — для всех 23 вариантов (`ALL_CLUSTERING_EVAL_VARIANTS`), готово.

**LLM-оценка** — сокращённый прогон (`scripts/run_cluster_llm_selection.py`):

1. `hdbscan_fine`, `hdbscan_medium` — label → coherence → distinctness (параллельно по кластерам,
   `--workers 6`).
2. Победитель по `coherence_weighted` (tie-break: `dup_pair_rate`).
3. Бейзлайны `kmeans_umap10_k{K}` при K победителя (979) и при K `hdbscan_medium` (242).

Итог: `results/summary_metrics.csv`, `results/selection_report.md`.

| variant | K | coherence_weighted | dup_pair_rate | entropy_norm | size_p50 | noise |
| --- | --- | --- | --- | --- | --- | --- |
| **hdbscan_fine** | **979** | **2.801** | 0.107 | 0.019 | 112 | 0.647 |
| hdbscan_medium | 242 | 2.416 | 0.010 | 0.032 | 580 | 0.358 |
| kmeans_umap10_k979 | 979 | 2.178 | 0.260 | 0.000 | 442 | 0.000 |
| kmeans_umap10_k242 | 242 | 1.868 | 0.033 | 0.000 | 1911 | 0.000 |

**Итог:** `hdbscan_fine` — лучший по coherence; `hdbscan_medium` ниже по coherence, но
существенно лучше по различимости (dup_pair_rate = 0.01). K-Means при K=979 хуже по coherence и
dup (0.26); при K=242 coherence ещё ниже (1.868), dup лучше (0.033), но хуже, чем у medium.

### Метакластеры (979 fine → semantic overlay)

Fine-темы (`hdbscan_fine`, K=979) объединены в метакластеры по эмбеддингам метадокументов
(multilingual-e5-base, префикс `query:`). LLM-оценка — `scripts/run_meta_llm_selection.py`:

- `meta_hdbscan_medium` (K=76)
- бейзлайн `meta_kmeans_umap10_k76`

Для метакластеров используются **отдельные** LLM-судьи (`metric_meta_coherence.py`,
`metric_meta_distinctness.py`, prompt `meta_coherence_v1` / `meta_distinctness_v1`): судья оценивает,
насколько fine-темы и репрезентативные статьи образуют одну **широкую исследовательскую область**,
а не узкую fine-тему (как в `coherence_strict_v2` для paper-кластеров). Вес `coherence_weighted` —
по числу статей fine-кластеров в метакластере (`hdbscan_fine/sizes.json`), не по числу fine-тем.

Итог: `results/meta_clustering/summary_metrics.csv`, `data/meta/meta_assignments.json`.

| variant | K | coherence_weighted | dup_pair_rate | entropy_norm | size_p50 | noise |
| --- | --- | --- | --- | --- | --- | --- |
| **meta_hdbscan_medium** | **76** | **2.708** | 0.000 | 0.330 | 10 | 0.114 |
| meta_kmeans_umap10_k76 | 76 | 2.417 | 0.000 | 0.000 | 12.5 | 0.000 |

**Итог:** `meta_hdbscan_medium` — победитель; k-means при том же K=76 ниже по meta-coherence.
Числа meta-coherence **не сопоставимы** напрямую с fine (2.801): другая шкала и постановка.

---

## 3.3. Author retrieval: семантика, граф и метакластеры

### Постановка и протокол оценки

Задача — **retrieval** потенциальных coauthors: по истории автора (≤ 2024) найти авторов из
того же split, которые станут новыми coauthors после 2024. Этот observed-сигнал неполный, поэтому
для итогового вывода вместе с observed-метриками используется LLM-оценка top-10 кандидатов.

- **Loss:** multi-positive **coauthor InfoNCE**; positives исключены из знаменателя.
- **Early stopping:** `val_ndcg@50`.
- **Онлайн-бюджет:** retrieval отдаёт top-10 кандидатов, затем LLM выбирает 3 финальные рекомендации.
- **Сравниваемые сигналы:** mean semantic baseline, Transformer, GraphSAGE по графу соавторства,
  Transformer+GraphSAGE.

### Конфигурации

Transformer-варианты: `d_model = 256`, 2 слоя, 4 heads, dropout 0.2, batch_size 64,
`n_negatives = 256`, `tau = 0.05`, `max_history = 20`, AdamW (`lr = 2e-4`,
`weight_decay = 0.05`). Лучший `author_retriever` выбран на epoch 7 по val nDCG@50;
`transformer_author_no_cluster` — на epoch 4.

GraphSAGE-варианты: 2 слоя, hidden_dim 256, dropout 0.2, `n_negatives = 256`, `tau = 0.05`,
`max_history = 20`, AdamW (`lr = 1e-3`, `weight_decay = 1e-2`). Граф содержит 35 954 авторских
узла и 464 752 направленных author-author ребра. В метакластерном варианте добавлены 76 meta-узлов
и 172 376 направленных author-meta/meta-author рёбер.

### Observed-результаты, test, K = 10

Источник: `results/retrieval/consistency/retrieval_metrics_at_k.csv`,
`results/retrieval/transformer_variants_metrics.csv`.

| method | Hit@10 | MRR@10 | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| mean_author_embedding | 0.0172 | 0.0059 | 0.0076 | 0.0148 |
| transformer_author_no_cluster | 0.0198 | 0.0071 | 0.0088 | 0.0173 |
| transformer_author_fine (`author_retriever`) | 0.0198 | 0.0073 | 0.0089 | 0.0168 |
| graphsage_author | 0.0182 | 0.0074 | 0.0087 | 0.0150 |
| graphsage_author_metacluster | 0.0203 | 0.0072 | 0.0087 | 0.0169 |
| **graphsage_transformer_author** | **0.0223** | **0.0086** | **0.0102** | **0.0186** |

### LLM-оценка, 40 test-users, K = 10

Источник: `results/retrieval/consistency/llm_threshold_metrics_at_k.csv`.

| method | Has3 >=2 | Selected3Mean | Hit =3 | MRR =3 | Has3 =3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| author_retriever | 0.875 | 2.425 | 0.625 | 0.491 | 0.450 |
| graphsage_author_metacluster | 0.775 | 2.367 | 0.700 | **0.541** | 0.425 |
| **graphsage_transformer_author** | **0.875** | **2.492** | **0.750** | 0.521 | **0.450** |

Вывод для текста: GraphSAGE можно включать как осмысленную проверку графового сигнала. Он даёт
качество, сопоставимое с Transformer, но текущие артефакты не доказывают значимый выигрыш от
кластерных узлов. Для метакластерного GraphSAGE против базового GraphSAGE paired bootstrap даёт
95% CI, включающие ноль: Hit@10 `[-0.002534; 0.006589]`, nDCG@10 `[-0.002221; 0.002326]`.

### Исправления pipeline (отладка)

До исправлений trainable retriever проигрывал mean baseline. Найдены две ошибки:

1. **Self-referential detached targets** в InfoNCE — positives/negatives кодировались с
   `detach=True`, градиент не обучал представления.
2. **Отсутствие якоря** — модель учила embedding с нуля, тогда как mean baseline напрямую
   использует качественные paper embeddings.

После symmetric loss + residual-from-mean модель стабильно обгоняет mean на AI-корпусе.

---

## 3.4. Иллюстрации для текста

| Файл | Содержание |
| --- | --- |
| `results/size_distribution.png` | Распределение размеров кластеров (fine / medium / k-means) |
| `figures/ui_map.png` | UI: карта метакластеров |
| `figures/ui_recommendations.png` | UI: панель рекомендаций |

---

## 3.5. Открытые пункты

1. **LLM-coherence** для FCM/GMM/coarse на AI-корпусе (не запускались; основной выбор уже по fine/medium).
2. **LLM reranking eval** на AI-корпусе и обновление UI-артефактов под 509k.
3. **Sweeps** (τ, capacity, learning curve) для `coauthor_infonce` на AI-корпусе.
