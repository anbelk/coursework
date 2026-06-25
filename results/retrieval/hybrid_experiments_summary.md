# Hybrid Graph Retrieval Experiments

Date: 2026-06-22.

Task: raw retrieval of potential coauthors before LLM reranking/evidence selection.
Evaluation split: `test`.
LLM evaluation: 40 users, top-10 raw nearest authors, prompt from `eval_retrieval_llm.py`.

## Methods

- `mean_author_embedding`: mean of paper embeddings in author history.
- `author_retriever`: Transformer over paper embeddings and fine-cluster membership vectors.
- `graphsage_author`: mean paper embedding + past coauthor graph.
- `graphsage_author_metacluster`: mean paper embedding + past coauthor graph + 76 meta-cluster nodes.
- `fusion_author_retriever_graphsage_author_a0p30`: concatenated score-equivalent fusion, 0.30 Transformer + 0.70 `graphsage_author`.
- `graphsage_transformer_author`: Transformer author embedding + past coauthor graph.
- `graphsage_transformer_author_metacluster`: Transformer author embedding + past coauthor graph + 76 meta-cluster nodes.

## Main Results

Full table: `results/retrieval/hybrid_summary.csv`.

| method | observed nDCG@10 | observed nDCG@50 | LLM nDCG@10 | LLM Precision@10 >=2 |
| --- | ---: | ---: | ---: | ---: |
| mean_author_embedding | 0.00761 | 0.01244 | 0.81466 | 0.6200 |
| author_retriever | 0.00892 | 0.01379 | 0.84358 | 0.6725 |
| graphsage_author | 0.00866 | 0.01426 | 0.75288 | 0.5525 |
| graphsage_author_metacluster | 0.00874 | 0.01431 | 0.86630 | 0.6500 |
| fusion_author_retriever_graphsage_author_a0p30 | 0.01084 | 0.01550 | 0.82962 | 0.6000 |
| graphsage_transformer_author | 0.01022 | 0.01684 | 0.84895 | 0.6800 |
| graphsage_transformer_author_metacluster | 0.00924 | 0.01548 | 0.85184 | 0.6525 |

## Interpretation

The strongest observed coauthor retrieval result is `graphsage_transformer_author`: it combines article-level Transformer features with the past coauthor graph. It improves over `author_retriever` on observed nDCG@10 and nDCG@50 and also slightly improves LLM nDCG@10, Precision@10 >=2, and mean LLM score.

The strongest LLM nDCG@10 remains `graphsage_author_metacluster`. This supports the thesis that meta-cluster information improves semantic ordering in raw retrieval. However, when Transformer author features are already used, adding meta-cluster nodes did not improve observed metrics and reduced LLM precision relative to `graphsage_transformer_author`.

Practical conclusion for the coursework: the best core method is `graphsage_transformer_author`, and the meta-cluster graph should be presented as a useful semantic ablation rather than as the final winner across all metrics.

## Additional Ablation

`graphsage_transformer_author_metacluster_mw005` repeats the Transformer-feature meta-graph with a stricter author--meta edge threshold (`meta_min_weight = 0.05` instead of `0.02`). It reduced author--meta edges from 86,188 to 84,525, but did not improve quality:

| method | observed nDCG@10 | observed nDCG@50 |
| --- | ---: | ---: |
| graphsage_transformer_author_metacluster_mw005 | 0.00924 | 0.01547 |

This suggests that the current issue is not only weak low-weight meta edges; the meta-node message passing itself is not yet adding useful signal on top of Transformer author features.

## Propagation Methods From Recent Graph Recommender Literature

Motivation: recent graph recommender papers often simplify the GNN and focus on propagation/contrastive smoothing rather than heavy nonlinear message passing. The tested family is close to LightGCN/LightGCL-style propagation and APPNP-style personalized propagation over the past coauthor graph, initialized with Transformer author embeddings.

Primary references checked during this cycle:

- LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation (2023).
- XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation (2022).
- GFormer: Graph Transformer for Recommendation (2023).
- SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation (2024).

Observed retrieval:

| method | selection note | observed nDCG@10 | observed nDCG@50 | LLM nDCG@10 | LLM Precision@10 >=2 |
| --- | --- | ---: | ---: | ---: | ---: |
| prop_lightgcn_l3_d1p0 | best by val nDCG@50 in propagation sweep | 0.01261 | 0.01838 | 0.81240 | 0.5900 |
| prop_appnp_s2_a0p1 | best propagation variant on test, diagnostic only | 0.01360 | 0.01962 | 0.82531 | 0.5575 |

These propagation methods substantially improve observed coauthor retrieval, but their LLM semantic quality is worse than `graphsage_transformer_author` and `graphsage_author_metacluster`. Interpretation: graph smoothing follows future collaboration topology better, but it over-smooths semantic author representations and brings less semantically precise raw candidates.
