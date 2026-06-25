# Cluster -> recommendation ablation

This file summarizes checks for whether clustering improves recommendation quality.

## Author retriever q-input ablation

`zero_q_at_inference` keeps the trained checkpoint fixed and replaces paper cluster-distribution inputs with zeros at inference. This is not a retrain-without-clusters experiment; it measures whether the trained model relies on cluster features at inference.

| split   | method           | cluster_mode        |   n_examples |   hit@10 |   mrr@10 |   ndcg@10 |   hit@50 |   mrr@50 |   ndcg@50 |   hit@100 |   mrr@100 |   ndcg@100 |   n_with_relevant |
|:--------|:-----------------|:--------------------|-------------:|---------:|---------:|----------:|---------:|---------:|----------:|----------:|----------:|-----------:|------------------:|
| val     | author_retriever | normal_q            |  2021.000000 | 0.029193 | 0.012449 |  0.015147 | 0.051460 | 0.013386 |  0.019942 |  0.064325 |  0.013574 |   0.022035 |        280.000000 |
| val     | author_retriever | zero_q_at_inference |  2021.000000 | 0.029688 | 0.012499 |  0.015290 | 0.051460 | 0.013393 |  0.019951 |  0.064325 |  0.013581 |   0.022043 |        280.000000 |
| test    | author_retriever | normal_q            |  1973.000000 | 0.019767 | 0.007315 |  0.008924 | 0.044095 | 0.008489 |  0.013793 |  0.058287 |  0.008697 |   0.015956 |        279.000000 |
| test    | author_retriever | zero_q_at_inference |  1973.000000 | 0.019767 | 0.007315 |  0.008924 | 0.044095 | 0.008489 |  0.013793 |  0.058287 |  0.008697 |   0.015955 |        279.000000 |

## Graph metacluster ablation: observed future coauthors

| comparison                                 | base                         | clustered                                |   K |   base_hit |   clustered_hit |   diff_hit |   base_mrr |   clustered_mrr |   diff_mrr |   base_ndcg |   clustered_ndcg |   diff_ndcg |   base_recall |   clustered_recall |   diff_recall |
|:-------------------------------------------|:-----------------------------|:-----------------------------------------|----:|-----------:|----------------:|-----------:|-----------:|----------------:|-----------:|------------:|-----------------:|------------:|--------------:|-------------------:|--------------:|
| mean_features_graphsage_metacluster        | graphsage_author             | graphsage_author_metacluster             |   5 |   0.011657 |        0.010137 |  -0.001521 |   0.006564 |        0.005854 |  -0.000710 |    0.006860 |         0.005946 |   -0.000914 |      0.009503 |           0.008743 |     -0.000760 |
| mean_features_graphsage_metacluster        | graphsage_author             | graphsage_author_metacluster             |  10 |   0.018246 |        0.020274 |   0.002027 |   0.007371 |        0.007211 |  -0.000160 |    0.008659 |         0.008738 |    0.000079 |      0.014994 |           0.016937 |      0.001943 |
| mean_features_graphsage_metacluster        | graphsage_author             | graphsage_author_metacluster             |  20 |   0.029397 |        0.030917 |   0.001521 |   0.008154 |        0.007957 |  -0.000197 |    0.011196 |         0.011138 |   -0.000058 |      0.024244 |           0.025764 |      0.001521 |
| transformer_features_graphsage_metacluster | graphsage_transformer_author | graphsage_transformer_author_metacluster |   5 |   0.015712 |        0.011657 |  -0.004055 |   0.007848 |        0.005879 |  -0.001968 |    0.008448 |         0.006319 |   -0.002129 |      0.013220 |           0.009672 |     -0.003548 |
| transformer_features_graphsage_metacluster | graphsage_transformer_author | graphsage_transformer_author_metacluster |  10 |   0.022301 |        0.021287 |  -0.001014 |   0.008642 |        0.007201 |  -0.001441 |    0.010218 |         0.009245 |   -0.000973 |      0.018584 |           0.018331 |     -0.000253 |
| transformer_features_graphsage_metacluster | graphsage_transformer_author | graphsage_transformer_author_metacluster |  20 |   0.036493 |        0.032945 |  -0.003548 |   0.009630 |        0.007989 |  -0.001642 |    0.013624 |         0.011884 |   -0.001740 |      0.031086 |           0.028214 |     -0.002872 |

## Graph metacluster ablation: LLM metrics
No pair with both base and clustered variants is available in stored LLM ratings.

## Paired bootstrap for metacluster graph ablation

Bootstrap is computed over test users at `K = 10`.

| comparison                                 | baseline                     | clustered                                | metric    |   mean_diff |   ci95_low |   ci95_high | significant_95 |
|:-------------------------------------------|:-----------------------------|:-----------------------------------------|:----------|------------:|-----------:|------------:|:---------------|
| mean_features_graphsage_metacluster        | graphsage_author             | graphsage_author_metacluster             | hit@10    |    0.002027 |  -0.002534 |    0.006589 | False          |
| mean_features_graphsage_metacluster        | graphsage_author             | graphsage_author_metacluster             | mrr@10    |   -0.000160 |  -0.002457 |    0.002165 | False          |
| mean_features_graphsage_metacluster        | graphsage_author             | graphsage_author_metacluster             | ndcg@10   |    0.000079 |  -0.002221 |    0.002326 | False          |
| mean_features_graphsage_metacluster        | graphsage_author             | graphsage_author_metacluster             | recall@10 |    0.001943 |  -0.001689 |    0.005913 | False          |
| transformer_features_graphsage_metacluster | graphsage_transformer_author | graphsage_transformer_author_metacluster | hit@10    |   -0.001014 |  -0.005081 |    0.003041 | False          |
| transformer_features_graphsage_metacluster | graphsage_transformer_author | graphsage_transformer_author_metacluster | mrr@10    |   -0.001441 |  -0.003325 |    0.000236 | False          |
| transformer_features_graphsage_metacluster | graphsage_transformer_author | graphsage_transformer_author_metacluster | ndcg@10   |   -0.000973 |  -0.002706 |    0.000674 | False          |
| transformer_features_graphsage_metacluster | graphsage_transformer_author | graphsage_transformer_author_metacluster | recall@10 |   -0.000253 |  -0.003675 |    0.003168 | False          |
