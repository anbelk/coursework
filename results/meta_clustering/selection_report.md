# Meta-Clustering Selection Report

Best variant: `meta_hdbscan_medium`

Size distribution plot: `size_distribution.png`

| variant                |   K |   coherence_weighted | coherence_proxy   |   dup_pair_rate | distinctness_proxy   |   mean_entropy_norm |   size_p50 |   top5_concentration |   noise_ratio | metrics_are_real   |
|:-----------------------|----:|---------------------:|:------------------|----------------:|:---------------------|--------------------:|-----------:|---------------------:|--------------:|:-------------------|
| meta_hdbscan_medium    |  76 |               2.7084 | False             |          0.0000 | False                |              0.3297 |    10.0000 |               0.1453 |        0.1144 | True               |
| meta_kmeans_umap10_k76 |  76 |               2.4170 | False             |          0.0000 | False                |              0.0000 |    12.5000 |               0.1307 |        0.0000 | True               |

Winner selection: highest `coherence_weighted` (tie-break: lower `dup_pair_rate`).
Proxy LLM metrics are excluded unless aggregate_meta.py is run with --allow-proxy.
