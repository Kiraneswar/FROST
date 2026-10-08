# Generated result tables

_Source run: `stage6_seed0_728fdb777ee6_20260912-160648`_

## Ablation ladder (accuracy, mean ± std)

| Stage | Acc mean | Acc std | F1 mean | F1 std | Δacc |
|---|---|---|---|---|---|
| 1_sync_ddil | 0.6543 | 0.1976 | 0.295 | 0.3875 |  |
| 2_+async | 0.6426 | 0.1792 | 0.2747 | 0.3535 | -0.0117 |
| 3_+staleness | 0.6204 | 0.1486 | 0.2237 | 0.2754 | -0.0222 |
| 4_+resource | 0.6204 | 0.1486 | 0.2237 | 0.2754 | 0.0 |
| 5_+compression | 0.6204 | 0.1486 | 0.2237 | 0.2754 | 0.0 |


## Byzantine robustness (accuracy, mean ± std)

| Attack | Aggregation | Acc mean | Acc std | F1 mean | F1 std |
|---|---|---|---|---|---|
| gaussian | mean | 0.5232 | 0.1377 | 0.4405 | 0.132 |
| gaussian | trimmed_mean | 0.5501 | 0.1214 | 0.3457 | 0.2828 |
| gaussian | median | 0.5904 | 0.1215 | 0.1031 | 0.2063 |
