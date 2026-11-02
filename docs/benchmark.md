## Detection summary

| scenario     |   n |   roc_auc |   detection_rate |   false_positive_rate |   mean_window_iou |
|:-------------|----:|----------:|-----------------:|----------------------:|------------------:|
| safety       | 200 |     0.960 |            0.860 |                 0.000 |             0.605 |
| productivity | 200 |     1.000 |            1.000 |                 0.000 |             0.498 |

## Sensitivity to gaming intensity

| scenario     |   intensity |   n |   roc_auc |   detection_rate |   false_positive_rate |   mean_window_iou |
|:-------------|------------:|----:|----------:|-----------------:|----------------------:|------------------:|
| productivity |       0.250 | 200 |     0.762 |            0.220 |                 0.000 |             0.461 |
| productivity |       0.500 | 200 |     0.924 |            0.790 |                 0.000 |             0.496 |
| productivity |       0.750 | 200 |     0.986 |            0.960 |                 0.000 |             0.498 |
| productivity |       1.000 | 200 |     1.000 |            1.000 |                 0.000 |             0.498 |
| safety       |       0.250 | 200 |     0.549 |            0.020 |                 0.000 |             0.391 |
| safety       |       0.500 | 200 |     0.705 |            0.210 |                 0.000 |             0.555 |
| safety       |       0.750 | 200 |     0.848 |            0.510 |                 0.000 |             0.594 |
| safety       |       1.000 | 200 |     0.960 |            0.860 |                 0.000 |             0.605 |
