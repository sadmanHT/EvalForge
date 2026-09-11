# Phase 05 golden metric fixture calculations

Fixture: `backend/tests/fixtures/phase5/golden_metrics.json`

These expected values were calculated independently from the implementation and are the correctness oracle for Phase 05 metric tests.

## Exact / hierarchical / top-3

There are four examples. Predictions 1, 3, and 4 have the exact root-cause code; prediction 2 predicts `n_plus_one_query` for a `database_connection_leak` truth.

- exact accuracy = `3 / 4 = 0.75`
- prediction 2 remains in category `database_behavior`, so hierarchical accuracy = `4 / 4 = 1.0`
- prediction 2 ranks the true `database_connection_leak` second, so top-3 accuracy = `4 / 4 = 1.0`

## Normalized multiclass Brier

Definition per example: `0.5 * sum_k (p_k - y_k)^2`.

For examples 1, 3, and 4, the true class has probability 0.70 and the other probabilities are 0.10/0.05/0.05/0.05/0.05. Each contributes:

`0.5 * ((0.70 - 1)^2 + 0.10^2 + 4 * 0.05^2) = 0.055`.

Example 2 assigns 0.60 to the wrong class and 0.20 to the true class, with four other classes at 0.05:

`0.5 * (0.60^2 + (0.20 - 1)^2 + 4 * 0.05^2) = 0.505`.

Mean normalized Brier:

`(0.055 + 0.505 + 0.055 + 0.055) / 4 = 0.1675`.

## ECE / reliability

With five fixed-width bins, all confidences (0.70, 0.60, 0.70, 0.70) fall into `[0.6, 0.8)`.

- bin count = 4
- bin accuracy = `3 / 4 = 0.75`
- mean confidence = `(0.70 + 0.60 + 0.70 + 0.70) / 4 = 0.675`
- absolute gap = `|0.75 - 0.675| = 0.075`
- ECE = `(4 / 4) * 0.075 = 0.075`

## Latency

Latencies are 100, 200, 300, 400 ms. R-7 interpolation uses `position = (n - 1)q`.

- p50: position 1.5 -> `200 + 0.5 * (300 - 200) = 250 ms`
- p95: position 2.85 -> `300 + 0.85 * (400 - 300) = 385 ms`

## Cost

Per-query marginal costs are $0.01, $0.02, $0.03, $0.04.

- total marginal = `$0.10`
- marginal mean/query = `$0.10 / 4 = $0.025`
- fixture upfront cost = `$0.10`
- amortized mean/query = `($0.10 + $0.10) / 4 = $0.05`

## Parse failures

All four golden predictions have `parse_status=OK`, so parse-failure rate is `0 / 4 = 0.0`.
