# Retrieval consistency report

## Target metric decision

Primary online budget: `top_n = 10` retrieval candidates scored by LLM, then `top_k = 3` final recommendations.

Primary offline proxy metrics:
- observed coauthor `Hit@10`: at least one future new coauthor appears in the LLM candidate set;
- observed coauthor `MRR@10`: rank of the first future new coauthor;
- observed coauthor `nDCG@10`: ordering of all observed future new coauthors;
- LLM `has_3_ge2@10`: enough candidates with LLM score >= 2 to fill three final recommendations;
- LLM `has_1_eq3@10` and `selected3_mean_score@10`: quality of the best candidates after LLM scoring.

`@5` and `@20` are diagnostic cost-sensitivity checks, not the main target.

## Method policy

Keep methods that test different information sources:
- `mean_author_embedding`: article semantics, no training;
- `author_retriever`: trained semantic author encoder;
- one simple graph method: coauthor graph / metacluster graph ablation;
- one combined method: semantic author vector plus graph message passing.

Do not foreground graph variants that only change implementation detail and do not produce a stable improvement.

## Current observed retrieval metrics

| split   | method                                   |   K |       hit |        mrr |       ndcg |     recall |   n_examples |   n_with_relevant |
|:--------|:-----------------------------------------|----:|----------:|-----------:|-----------:|-----------:|-------------:|------------------:|
| test    | mean_author_embedding                    |   5 | 0.00963   | 0.00486569 | 0.00541871 | 0.00848961 |         1973 |               279 |
| test    | mean_author_embedding                    |  10 | 0.0172326 | 0.00592302 | 0.00760995 | 0.0148251  |         1973 |               279 |
| test    | mean_author_embedding                    |  20 | 0.025849  | 0.00647802 | 0.00944444 | 0.021752   |         1973 |               279 |
| test    | author_retriever                         |   5 | 0.0101368 | 0.00609056 | 0.00633223 | 0.00899645 |         1973 |               279 |
| test    | author_retriever                         |  10 | 0.0197669 | 0.00731502 | 0.00892383 | 0.016768   |         1973 |               279 |
| test    | author_retriever                         |  20 | 0.0283832 | 0.00796889 | 0.0108294  | 0.0234837  |         1973 |               279 |
| test    | graphsage_author                         |   5 | 0.0116574 | 0.00656361 | 0.00686032 | 0.00950329 |         1973 |               279 |
| test    | graphsage_author                         |  10 | 0.0182463 | 0.00737134 | 0.00865899 | 0.0149941  |         1973 |               279 |
| test    | graphsage_author                         |  20 | 0.0293969 | 0.00815384 | 0.011196   | 0.024244   |         1973 |               279 |
| test    | graphsage_author_metacluster             |   5 | 0.0101368 | 0.00585403 | 0.00594609 | 0.00874303 |         1973 |               279 |
| test    | graphsage_author_metacluster             |  10 | 0.0202737 | 0.00721144 | 0.0087378  | 0.016937   |         1973 |               279 |
| test    | graphsage_author_metacluster             |  20 | 0.0309174 | 0.00795669 | 0.0111378  | 0.0257645  |         1973 |               279 |
| test    | graphsage_transformer_author             |   5 | 0.0157121 | 0.00784761 | 0.00844807 | 0.0132201  |         1973 |               279 |
| test    | graphsage_transformer_author             |  10 | 0.0223011 | 0.00864207 | 0.0102176  | 0.0185842  |         1973 |               279 |
| test    | graphsage_transformer_author             |  20 | 0.0364926 | 0.00963037 | 0.0136239  | 0.0310863  |         1973 |               279 |
| test    | graphsage_transformer_author_metacluster |   5 | 0.0116574 | 0.00587937 | 0.00631861 | 0.00967224 |         1973 |               279 |
| test    | graphsage_transformer_author_metacluster |  10 | 0.0212874 | 0.00720139 | 0.00924494 | 0.0183308  |         1973 |               279 |
| test    | graphsage_transformer_author_metacluster |  20 | 0.0329448 | 0.00798883 | 0.0118842  | 0.0282142  |         1973 |               279 |

## Pairwise signal correlation

| method_left                  | method_right                             |   n_users |   n_common_candidates |   mean_score_pearson |   mean_rank_spearman |   top5_overlap_rate |   top5_jaccard |   top10_overlap_rate |   top10_jaccard |   top20_overlap_rate |   top20_jaccard |
|:-----------------------------|:-----------------------------------------|----------:|----------------------:|---------------------:|---------------------:|--------------------:|---------------:|---------------------:|----------------:|---------------------:|----------------:|
| mean_author_embedding        | author_retriever                         |      1973 |                  3569 |             0.890299 |             0.929669 |            0.753877 |       0.6402   |             0.770705 |        0.646946 |             0.782108 |        0.655171 |
| mean_author_embedding        | graphsage_author                         |      1973 |                  3569 |             0.712833 |             0.743464 |            0.391688 |       0.277637 |             0.416371 |        0.289238 |             0.441308 |        0.30603  |
| mean_author_embedding        | graphsage_author_metacluster             |      1973 |                  3569 |             0.748981 |             0.779023 |            0.628687 |       0.502757 |             0.656766 |        0.521454 |             0.666929 |        0.526994 |
| mean_author_embedding        | graphsage_transformer_author             |      1973 |                  3569 |             0.52301  |             0.531623 |            0.340497 |       0.234141 |             0.364927 |        0.243892 |             0.387101 |        0.256592 |
| mean_author_embedding        | graphsage_transformer_author_metacluster |      1973 |                  3569 |             0.597683 |             0.617855 |            0.452306 |       0.326974 |             0.481196 |        0.342621 |             0.493664 |        0.347507 |
| author_retriever             | graphsage_author                         |      1973 |                  3569 |             0.775236 |             0.806103 |            0.395033 |       0.28216  |             0.423416 |        0.296022 |             0.451394 |        0.315107 |
| author_retriever             | graphsage_author_metacluster             |      1973 |                  3569 |             0.817162 |             0.813775 |            0.717993 |       0.607298 |             0.736797 |        0.616267 |             0.749493 |        0.62514  |
| author_retriever             | graphsage_transformer_author             |      1973 |                  3569 |             0.591611 |             0.563137 |            0.399595 |       0.283021 |             0.42296  |        0.291267 |             0.44777  |        0.307234 |
| author_retriever             | graphsage_transformer_author_metacluster |      1973 |                  3569 |             0.649266 |             0.636566 |            0.522859 |       0.391417 |             0.545666 |        0.402482 |             0.562874 |        0.412677 |
| graphsage_author             | graphsage_author_metacluster             |      1973 |                  3569 |             0.673487 |             0.694764 |            0.436391 |       0.31356  |             0.469894 |        0.334972 |             0.496249 |        0.352628 |
| graphsage_author             | graphsage_transformer_author             |      1973 |                  3569 |             0.53134  |             0.531285 |            0.353573 |       0.245744 |             0.384896 |        0.261995 |             0.410061 |        0.276636 |
| graphsage_author             | graphsage_transformer_author_metacluster |      1973 |                  3569 |             0.576748 |             0.577854 |            0.414901 |       0.295968 |             0.442777 |        0.309406 |             0.466092 |        0.323885 |
| graphsage_author_metacluster | graphsage_transformer_author             |      1973 |                  3569 |             0.737731 |             0.697064 |            0.450177 |       0.324443 |             0.479878 |        0.339768 |             0.508667 |        0.359911 |
| graphsage_author_metacluster | graphsage_transformer_author_metacluster |      1973 |                  3569 |             0.826218 |             0.799389 |            0.600203 |       0.466353 |             0.627217 |        0.482623 |             0.643284 |        0.493278 |
| graphsage_transformer_author | graphsage_transformer_author_metacluster |      1973 |                  3569 |             0.899639 |             0.880232 |            0.606386 |       0.472375 |             0.64146  |        0.497021 |             0.66964  |        0.521309 |

## Paired bootstrap versus baseline

| baseline         | method                                   | metric   |    mean_diff |    ci95_low |    ci95_high | significant_95   |   n_examples |   n_boot |
|:-----------------|:-----------------------------------------|:---------|-------------:|------------:|-------------:|:-----------------|-------------:|---------:|
| author_retriever | mean_author_embedding                    | hit@10   | -0.00253421  | -0.00658895 |  0.00101368  | False            |         1973 |     2000 |
| author_retriever | mean_author_embedding                    | mrr@10   | -0.00139201  | -0.00263197 | -0.000309853 | True             |         1973 |     2000 |
| author_retriever | mean_author_embedding                    | ndcg@10  | -0.00131387  | -0.0027081  | -3.61119e-05 | True             |         1973 |     2000 |
| author_retriever | graphsage_author                         | hit@10   | -0.00152053  | -0.00558794 |  0.00254688  | False            |         1973 |     2000 |
| author_retriever | graphsage_author                         | mrr@10   |  5.63158e-05 | -0.00198109 |  0.00232181  | False            |         1973 |     2000 |
| author_retriever | graphsage_author                         | ndcg@10  | -0.000264834 | -0.0022393  |  0.00175939  | False            |         1973 |     2000 |
| author_retriever | graphsage_author_metacluster             | hit@10   |  0.000506842 | -0.00304105 |  0.00405474  | False            |         1973 |     2000 |
| author_retriever | graphsage_author_metacluster             | mrr@10   | -0.000103581 | -0.0012433  |  0.00116278  | False            |         1973 |     2000 |
| author_retriever | graphsage_author_metacluster             | ndcg@10  | -0.000186022 | -0.00136173 |  0.000919585 | False            |         1973 |     2000 |
| author_retriever | graphsage_transformer_author             | hit@10   |  0.00253421  | -0.00253421 |  0.00760264  | False            |         1973 |     2000 |
| author_retriever | graphsage_transformer_author             | mrr@10   |  0.00132704  | -0.00077657 |  0.00335653  | False            |         1973 |     2000 |
| author_retriever | graphsage_transformer_author             | ndcg@10  |  0.00129373  | -0.00124915 |  0.00372606  | False            |         1973 |     2000 |
| author_retriever | graphsage_transformer_author_metacluster | hit@10   |  0.00152053  | -0.00304105 |  0.00609478  | False            |         1973 |     2000 |
| author_retriever | graphsage_transformer_author_metacluster | mrr@10   | -0.000113637 | -0.00211151 |  0.00167199  | False            |         1973 |     2000 |
| author_retriever | graphsage_transformer_author_metacluster | ndcg@10  |  0.000321119 | -0.00193668 |  0.00244993  | False            |         1973 |     2000 |

## Existing LLM ratings re-aggregated at K

These rows reuse the existing `results/retrieval/llm_eval/ratings.csv`; they do not add new LLM calls.

| method                       |   K |   n_users |   llm_ndcg |   llm_relevant_rate_ge2 |   llm_mean_score |   llm_has_1_ge2 |   llm_has_3_ge2 |   llm_has_1_eq3 |   llm_selected3_mean_score |
|:-----------------------------|----:|----------:|-----------:|------------------------:|-----------------:|----------------:|----------------:|----------------:|---------------------------:|
| author_retriever             |   5 |        40 |   0.861886 |                  0.71   |           1.85   |           0.925 |           0.725 |           0.575 |                    2.18333 |
| author_retriever             |  10 |        40 |   0.843585 |                  0.6725 |           1.7325 |           0.975 |           0.875 |           0.625 |                    2.425   |
| graphsage_author_metacluster |   5 |        40 |   0.9115   |                  0.705  |           1.85   |           0.875 |           0.725 |           0.625 |                    2.19167 |
| graphsage_author_metacluster |  10 |        40 |   0.866302 |                  0.65   |           1.6825 |           0.925 |           0.775 |           0.7   |                    2.36667 |
| graphsage_transformer_author |   5 |        40 |   0.898661 |                  0.715  |           1.86   |           0.925 |           0.75  |           0.65  |                    2.2     |
| graphsage_transformer_author |  10 |        40 |   0.848955 |                  0.68   |           1.745  |           0.975 |           0.875 |           0.75  |                    2.49167 |
| prop_appnp_s2_a0p1           |   5 |        40 |   0.850206 |                  0.62   |           1.58   |           0.875 |           0.55  |           0.6   |                    1.95833 |
| prop_appnp_s2_a0p1           |  10 |        40 |   0.825307 |                  0.5575 |           1.4425 |           0.9   |           0.775 |           0.675 |                    2.26667 |
| prop_lightgcn_l3_d1p0        |   5 |        40 |   0.82715  |                  0.63   |           1.63   |           0.875 |           0.625 |           0.575 |                    2.00833 |
| prop_lightgcn_l3_d1p0        |  10 |        40 |   0.812398 |                  0.59   |           1.525  |           0.9   |           0.75  |           0.7   |                    2.26667 |
