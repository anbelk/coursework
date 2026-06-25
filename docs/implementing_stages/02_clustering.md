# Глава 2.3. Кластеризация научных публикаций

## Pipeline кластеризации

Кластеризация построена в стиле BERTopic:

```text
paper embeddings
  → UMAP, n_components = 10
  → HDBSCAN
  → c-TF-IDF
  → MMR
  → LLM labels
  → metrics: coherence, duplicate pairs, noise, size distribution
```

HDBSCAN выбран как основной метод, потому что число тем заранее неизвестно, плотность научных
областей неодинакова, а фоновые статьи лучше пометить шумом, чем насильно присвоить ближайшей теме.

## Выбранный fine-вариант

В PDF используется `hdbscan_fine`:

| Параметр | Значение |
| --- | ---: |
| Число fine-тем | 979 |
| `min_cluster_size` | 10 |
| `min_samples` | 5 |
| `cluster_selection_method` | `leaf` |
| `size_p50` | 112 |
| `noise_ratio` | 0.647 |

Главный компромисс: `hdbscan_fine` даёт максимально когерентные и детальные темы, но оставляет
большую долю шума. Это приемлемо для карты: лучше иметь чистые темы и отдельный фон, чем размывать
темы ради полного покрытия.

## Сравнение с бейзлайнами

Для проверки добавлены наивные бейзлайны:

- K-Means на том же UMAP-10 пространстве и с тем же числом кластеров;
- random assignment с тем же числом кластеров.

Итоговые источники:

- `results/summary_metrics.csv`;
- `results/summary_metrics_quant.csv`;
- `results/selection_report.md`;
- `results/final_tables/clustering_intrinsic.csv`.

## Интерпретация тем

Для каждой темы строятся:

- характерные термины через c-TF-IDF;
- разнообразный набор терминов через MMR;
- LLM-название по representative papers и terms.

Эти названия используются в карте и в метадокументах для построения метакластеров.
