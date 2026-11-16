# Frozen formal metric specification

The machine-readable authority is `data/week6/frozen_protocol/formal_metric_spec.json`. Primary analyses retain every scheduled episode—including model, runtime, and evaluator errors—in the denominator. A separately labelled error-free sensitivity analysis may be reported but cannot replace the primary result.

Frozen primary outcomes are targeted attack success, confidentiality violation, integrity violation, authorization violation, replay success/detection, audit-tamper detection, benign task success, utility under attack, false blocks, runtime/model failure, and latency/token/storage/throughput/approval costs.

Proportions use Wilson 95% intervals. Matched binary outcomes use the exact discordant-pair McNemar/binomial procedure. Paired cost differences use a deterministic 10,000-sample bootstrap with seed `20261009`.

All security outcomes are deterministic functions of final state and retained signed evidence. The semantic judge is disabled for the frozen security protocol. Its frozen schema and settings exist only to constrain a future genuinely semantic quality analysis.
