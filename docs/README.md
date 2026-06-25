# Курсовая работа: AI Coauthor

Документация проекта **AI Coauthor** — системы семантического анализа научного пространства,
кластеризации публикаций и рекомендации потенциальных соавторов с объяснениями на основе LLM.

Главный текст работы находится в LaTeX: `main.tex` и `sections/*.tex`. Собранный PDF:
`build/main.pdf`. Документы в этом каталоге — вспомогательные заметки и черновики к главам; при
расхождении итоговым источником считается PDF и таблицы в `results/final_tables/`.

Все числовые показатели, которые попали в PDF, взяты из реальных артефактов запусков (`results/`,
`data/`, `models/`), а не придуманы.

## Навигация

### Описание проекта (`project_description/`)

- [01. Введение, цели и задачи](project_description/01_introduction.md)
- [02. Архитектура системы (глава 2.1)](project_description/02_architecture.md)

### Этапы реализации (`implementing_stages/`)

- [01. Построение embedding статей (2.2)](implementing_stages/01_embeddings.md)
- [02. Кластеризация научных публикаций (2.3)](implementing_stages/02_clustering.md)
- [03. Метакластеры — семантический overlay (2.4)](implementing_stages/03_meta_clusters.md)
- [04. Рекомендательная система: author retrieval (2.5)](implementing_stages/04_author_retrieval.md)
- [05. LLM reranking и генерация объяснений (2.6)](implementing_stages/05_llm_reranking.md)
- [06. Веб-приложение (2.7)](implementing_stages/06_web_app.md)

### Эксперименты и выводы

- [Глава 3. Эксперименты](experiments.md)
- [Заключение и ограничения](conclusion.md)
- [Метрики, методы и консистентный pipeline](metrics_methods_pipeline.md)

## Рекомендуемая структура текста курсовой

1. **Введение** — актуальность, цель, задачи (см. `project_description/01_introduction.md`).
2. **Глава 1. Обзор предметной области** — научные графы, тематическое моделирование,
   эмбеддинги текстов, рекомендательные системы для соавторства, LLM-as-a-judge.
3. **Глава 2. Архитектура и реализация системы AI Coauthor**
   - 2.1 Общая архитектура.
   - 2.2 Построение embedding статей.
   - 2.3 Кластеризация научных публикаций.
   - 2.4 Метакластеры.
   - 2.5 Рекомендательная система.
   - 2.6 LLM reranking.
   - 2.7 Веб-приложение.
4. **Глава 3. Эксперименты** — датасет, сравнение алгоритмов кластеризации,
   retrieval evaluation, LLM evaluation, демонстрация интерфейса.
5. **Заключение** — полученные результаты и ограничения.

## Готовые изображения для вставки в текст

Эти файлы уже сгенерированы и пригодны для вставки в курсовую:

| Файл | Что показывает | Куда вставить |
| --- | --- | --- |
| `figures/size_distribution.png` | Распределение размеров fine-кластеров | Глава 2.3 / 3.2 |
| `figures/meta_size_distribution.png` | Распределение размеров метакластеров | Глава 2.4 / 3.2 |
| `figures/val_ndcg10.png` | Валидационная кривая Transformer retrieval | Глава 2.5 |
| `figures/ui_map.png` | Карта научного пространства | Глава 2.7 |
| `figures/ui_recommendations.png` | Окно рекомендаций | Глава 2.7 |

Изображения, которые нужно подготовить отдельно (скриншоты и схемы), перечислены в
[списке необходимых иллюстраций](implementing_stages/06_web_app.md#скриншоты-для-курсовой)
и в начале соответствующих глав.

## Источник числовых данных

| Показатель | Файл-источник |
| --- | --- |
| Выбор fine-кластеризации | `results/selection_report.md`, `results/summary_metrics.csv` |
| Выбор метакластеризации | `results/meta_clustering/selection_report.md` |
| Итоговые таблицы для PDF | `results/final_tables/tables.md` |
| Retrieval-метрики | `results/retrieval/consistency/retrieval_metrics_at_k.csv`, `results/retrieval/transformer_variants_metrics.csv` |
| LLM-оценка рекомендаций | `results/retrieval/consistency/llm_threshold_metrics_at_k.csv` |
| Диагностика роли кластеризации | `results/retrieval/cluster_ablation/summary.md` |
