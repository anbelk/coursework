# BERTopic Clustering Selection Report

Best variant: `hdbscan_fine`

Size distribution plot: `size_distribution.png`

| variant            |   K |   coherence_weighted |   dup_pair_rate |   mean_entropy_norm |   size_p50 |   top5_concentration |   noise_ratio |
|:-------------------|----:|---------------------:|----------------:|--------------------:|-----------:|---------------------:|--------------:|
| hdbscan_fine       | 979 |               2.8010 |          0.1067 |              0.0193 |   112.0000 |               0.0323 |        0.6468 |
| hdbscan_medium     | 242 |               2.4157 |          0.0100 |              0.0316 |   580.5000 |               0.2534 |        0.3577 |
| kmeans_umap10_k979 | 979 |               2.1784 |          0.2600 |              0.0000 |   441.5000 |               0.0210 |        0.0000 |
| kmeans_umap10_k242 | 242 |               1.8683 |          0.0333 |              0.0000 |  1910.5000 |               0.0768 |        0.0000 |

Winner selection: highest `coherence_weighted` (tie-break: lower `dup_pair_rate`).
Baselines (`random_umap10`, `kmeans_umap10`) are included for comparison but excluded from winner selection.
