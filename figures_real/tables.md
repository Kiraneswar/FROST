# FedZD — Real ToN_IoT + UNSW-NB15 results (5 seeds)

_Real data. ToN_IoT subsampled to 50K rows on CPU, 5 seeds; scale to full 461K on GPU for camera-ready._

## ToN_IoT — fusion weight sweep (overall detection, seed 0)

| Weights | Acc | Prec | Recall | F1 | Zero-day recall |
|---|---|---|---|---|---|
| classifier_only | 0.8607 | 0.7543 | 0.8185 | 0.7851 | 0.4917 |
| zaid_best | 0.8614 | 0.7546 | 0.8212 | 0.7865 | 0.5042 |
| equal | 0.8525 | 0.7275 | 0.8403 | 0.7798 | 0.6792 |
| ae_heavy | 0.8511 | 0.7326 | 0.8203 | 0.774 | 0.5 |
| proposed_A2C1U1 | 0.8484 | 0.7268 | 0.8207 | 0.7709 | 0.5 |
| proposed_A1_5C1_5U1 | 0.8528 | 0.7354 | 0.8221 | 0.7764 | 0.5125 |

## ToN_IoT — per-family recall (zero-day = withheld)

| Family | Zero-day | Recall | n |
|---|---|---|---|
| scanning | False | 1.0 | 330 |
| backdoor | False | 0.9971 | 341 |
| password | False | 0.9904 | 312 |
| injection | False | 0.9849 | 331 |
| xss | False | 0.9844 | 320 |
| ddos | True | 0.9625 | 80 |
| dos | True | 0.95 | 80 |
| ransomware | False | 0.1975 | 324 |
| mitm | True | 0.125 | 80 |

## Ablation ladder — 5-seed mean ± std

| Stage | Acc | ± | F1 | Δacc |
|---|---|---|---|---|
| 1_sync_ddil | 0.815 | 0.0757 | 0.7385 |  |
| 2_+async | 0.7418 | 0.0933 | 0.6846 | -0.0731 |
| 3_+staleness | 0.7417 | 0.0928 | 0.6832 | -0.0001 |
| 4_+resource | 0.7293 | 0.079 | 0.6542 | -0.0124 |
| 5_+compression | 0.7293 | 0.079 | 0.6542 | -0.0 |

## Byzantine robustness (Gaussian poison) — 5-seed mean ± std

| Attack | Rule | Acc | ± |
|---|---|---|---|
| gaussian | mean | 0.6557 | 0.0416 |
| gaussian | trimmed_mean | 0.6984 | 0.0248 |
| gaussian | median | 0.6281 | 0.1713 |

## Cross-domain transfer ToN_IoT → UNSW-NB15

| Config | Acc | Prec | Recall | F1 |
|---|---|---|---|---|
| classifier_only | 0.6343 | 0.6228 | 0.8514 | 0.7194 |
| fusion_A1C1 | 0.5881 | 0.5811 | 0.9027 | 0.707 |

_ZAID reference: ToN_IoT zero-day 83.16/97.69/67.86/79.40; UNSW transfer 71.64/75.99/62.75/67.94 (acc/prec/rec/F1)._
