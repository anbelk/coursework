# All retrieval metrics

Observed coauthor metrics use future coauthor labels, not LLM scores.

## Observed future-coauthor metrics @5

| method                                   |      hit |      mrr |     ndcg |   recall |   n_examples |   n_with_relevant |
|:-----------------------------------------|---------:|---------:|---------:|---------:|-------------:|------------------:|
| mean_author_embedding                    | 0.009630 | 0.004866 | 0.005419 | 0.008490 |         1973 |               279 |
| author_retriever                         | 0.010137 | 0.006091 | 0.006332 | 0.008996 |         1973 |               279 |
| graphsage_author                         | 0.011657 | 0.006564 | 0.006860 | 0.009503 |         1973 |               279 |
| graphsage_author_metacluster             | 0.010137 | 0.005854 | 0.005946 | 0.008743 |         1973 |               279 |
| graphsage_transformer_author             | 0.015712 | 0.007848 | 0.008448 | 0.013220 |         1973 |               279 |
| graphsage_transformer_author_metacluster | 0.011657 | 0.005879 | 0.006319 | 0.009672 |         1973 |               279 |


## Observed future-coauthor metrics @10

| method                                   |      hit |      mrr |     ndcg |   recall |   n_examples |   n_with_relevant |
|:-----------------------------------------|---------:|---------:|---------:|---------:|-------------:|------------------:|
| mean_author_embedding                    | 0.017233 | 0.005923 | 0.007610 | 0.014825 |         1973 |               279 |
| author_retriever                         | 0.019767 | 0.007315 | 0.008924 | 0.016768 |         1973 |               279 |
| graphsage_author                         | 0.018246 | 0.007371 | 0.008659 | 0.014994 |         1973 |               279 |
| graphsage_author_metacluster             | 0.020274 | 0.007211 | 0.008738 | 0.016937 |         1973 |               279 |
| graphsage_transformer_author             | 0.022301 | 0.008642 | 0.010218 | 0.018584 |         1973 |               279 |
| graphsage_transformer_author_metacluster | 0.021287 | 0.007201 | 0.009245 | 0.018331 |         1973 |               279 |


## Observed future-coauthor metrics @20

| method                                   |      hit |      mrr |     ndcg |   recall |   n_examples |   n_with_relevant |
|:-----------------------------------------|---------:|---------:|---------:|---------:|-------------:|------------------:|
| mean_author_embedding                    | 0.025849 | 0.006478 | 0.009444 | 0.021752 |         1973 |               279 |
| author_retriever                         | 0.028383 | 0.007969 | 0.010829 | 0.023484 |         1973 |               279 |
| graphsage_author                         | 0.029397 | 0.008154 | 0.011196 | 0.024244 |         1973 |               279 |
| graphsage_author_metacluster             | 0.030917 | 0.007957 | 0.011138 | 0.025764 |         1973 |               279 |
| graphsage_transformer_author             | 0.036493 | 0.009630 | 0.013624 | 0.031086 |         1973 |               279 |
| graphsage_transformer_author_metacluster | 0.032945 | 0.007989 | 0.011884 | 0.028214 |         1973 |               279 |


# LLM relevance metrics

LLM metrics use LLM scores in {0,1,2,3}; relevance threshold is score >= 2. Only @5 and @10 are available because the stored LLM ratings contain top-10 candidates.

## LLM relevance metrics @5

| method                       |   llm_hit_ge2 |   llm_mrr_ge2 |   llm_ndcg |   llm_relevant_rate_ge2 |   llm_mean_score |   llm_has_3_ge2 |   llm_has_1_eq3 |   llm_selected3_mean_score |
|:-----------------------------|--------------:|--------------:|-----------:|------------------------:|-----------------:|----------------:|----------------:|---------------------------:|
| author_retriever             |      0.925000 |      0.827917 |   0.861886 |                0.710000 |         1.850000 |        0.725000 |        0.575000 |                   2.183333 |
| graphsage_author_metacluster |      0.875000 |      0.839583 |   0.911500 |                0.705000 |         1.850000 |        0.725000 |        0.625000 |                   2.191667 |
| graphsage_transformer_author |      0.925000 |      0.821667 |   0.898661 |                0.715000 |         1.860000 |        0.750000 |        0.650000 |                   2.200000 |
| prop_appnp_s2_a0p1           |      0.875000 |      0.777083 |   0.850206 |                0.620000 |         1.580000 |        0.550000 |        0.600000 |                   1.958333 |
| prop_lightgcn_l3_d1p0        |      0.875000 |      0.747500 |   0.827150 |                0.630000 |         1.630000 |        0.625000 |        0.575000 |                   2.008333 |


## LLM relevance metrics @10

| method                       |   llm_hit_ge2 |   llm_mrr_ge2 |   llm_ndcg |   llm_relevant_rate_ge2 |   llm_mean_score |   llm_has_3_ge2 |   llm_has_1_eq3 |   llm_selected3_mean_score |
|:-----------------------------|--------------:|--------------:|-----------:|------------------------:|-----------------:|----------------:|----------------:|---------------------------:|
| author_retriever             |      0.975000 |      0.834613 |   0.843585 |                0.672500 |         1.732500 |        0.875000 |        0.625000 |                   2.425000 |
| graphsage_author_metacluster |      0.925000 |      0.845933 |   0.866302 |                0.650000 |         1.682500 |        0.775000 |        0.700000 |                   2.366667 |
| graphsage_transformer_author |      0.975000 |      0.827569 |   0.848955 |                0.680000 |         1.745000 |        0.875000 |        0.750000 |                   2.491667 |
| prop_appnp_s2_a0p1           |      0.900000 |      0.780655 |   0.825307 |                0.557500 |         1.442500 |        0.775000 |        0.675000 |                   2.266667 |
| prop_lightgcn_l3_d1p0        |      0.900000 |      0.751667 |   0.812398 |                0.590000 |         1.525000 |        0.750000 |        0.700000 |                   2.266667 |
