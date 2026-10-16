# AM-0 diagnostic results

**Decision: do not scale AM-0 as a new architecture.** The exact mathematical reduction failed the strong novelty gate before performance measurements. The bounded diagnostic below tests whether this constrained feature learner nevertheless has a useful predictive tradeoff. It does not validate biological prediction or ASI.

Protocol: [AM-0/1](PROTOCOL.md). Rule: [conservative assembly exchange](LEARNING_RULE.md). Algebra: [exact coefficient/occupancy reduction](MATHEMATICAL_AUDIT.md).

## What was run

- 84 development fits: seven trained methods, three rates, four independent development systems.
- 15 held-out systems: five independently sampled systems in each of three abstract response families. The delayed-feedback family was absent during development.
- 270 evaluation records: nine methods × two budget settings × 15 systems; 240 trained fits and 30 untrained controls. These records are not 270 independent systems.
- Each system supplies 1,408 unique training labels and 880 evaluation predictions. Evaluation uses new histories, including simultaneous high-high stimulus values excluded during training.
- Fixed history encoding and one-step predictions using observed past response. No biological sequences, real cellular dataset, long-horizon autonomous rollout, learned compartment routing, selective transfer or causal discovery.

## Main interpretation

AM-0 learns useful associations: it beats the untrained and shuffled-outcome controls in every evaluated system. That establishes learning in this synthetic setting, not novelty.

The full quadratic LMS control beats AM-0 in **all 15 systems under the time ceiling**. AM-0's mean error is approximately 2.72 times the comparator's error on adaptive response, 2.79 times on resource response, and 1.80 times on delayed feedback (AM-0 error divided by quadratic LMS error). One-pass outcomes are mixed; AM-0's adaptive-family mean is slightly better, while the resource and delayed means are worse. The conventional learner trains faster and uses less persistent state in this implementation.

Pooling often improves on the no-pool version, but that narrower regularization effect does not establish a new learning principle. Adaptive membership does not give a consistent advantage across all five systems in any family under the time ceiling. There is no basis here for a GPU scaling run.

## Predictive measurements

Numbers are mean evaluation normalized MSE across five systems per family; lower is better. Normalization uses that system's training-response variance with a floor of 0.01. These are errors, not accuracy percentages. All rates were selected on development validation and frozen before evaluation.

### One pass: same labels and update count, measured runtime

| Method | Adaptive response | Resource response | Delayed feedback |
|---|---:|---:|---:|
| assembly | 0.01961 | 0.01221 | 0.03663 |
| assembly_fixed | 0.02147 | 0.01100 | 0.03942 |
| assembly_no_pool | 0.01984 | 0.01285 | 0.03622 |
| adaptive_lms | 0.02611 | 0.01683 | 0.02746 |
| quadratic_lms | 0.02050 | 0.01050 | 0.03039 |
| history_lms | 0.04903 | 0.02265 | 0.02569 |
| current_lms | 0.32069 | 0.29468 | 0.11775 |
| assembly_shuffled | 1.34365 | 1.04045 | 1.31507 |
| assembly_untrained | 1.19179 | 1.34418 | 0.94591 |

### Common 0.75-second training ceiling: same finite dataset, replay allowed

| Method | Adaptive response | Resource response | Delayed feedback |
|---|---:|---:|---:|
| assembly | 0.00842 | 0.00650 | 0.02680 |
| assembly_fixed | 0.01060 | 0.00690 | 0.02931 |
| assembly_no_pool | 0.01093 | 0.00921 | 0.04476 |
| adaptive_lms | 0.00320 | 0.00296 | 0.01417 |
| quadratic_lms | 0.00310 | 0.00233 | 0.01489 |
| history_lms | 0.01443 | 0.00524 | 0.01316 |
| current_lms | 0.33074 | 0.28933 | 0.11417 |
| assembly_shuffled | 1.16772 | 1.14575 | 0.63558 |
| assembly_untrained | 1.19179 | 1.34418 | 0.94591 |

### Paired effect of AM-0 versus full quadratic LMS

Positive improvement favors AM-0. Wins/losses are across five independent systems; ties use relative tolerance 1e-10. These small counts do not support a population-wide superiority claim.

| Setting / family | Median relative error improvement | Range | Wins / ties / losses |
|---|---:|---:|---:|
| one_pass / adaptive | +1.0% | -22.5% to +31.5% | 3 / 0 / 2 |
| one_pass / resource | -11.8% | -68.0% to +5.2% | 1 / 0 / 4 |
| one_pass / delayed | +3.6% | -106.7% to +18.6% | 3 / 0 / 2 |
| time_cap / adaptive | -129.9% | -316.4% to -43.5% | 0 / 0 / 5 |
| time_cap / resource | -221.7% | -271.7% to -49.1% | 0 / 0 / 5 |
| time_cap / delayed | -70.8% | -160.7% to -48.4% | 0 / 0 / 5 |

The predeclared predictive gate **failed**. It required a 10% median error improvement against both adaptive and full quadratic LMS in every family and both settings, with complete data exposure. Performance cannot override the separate novelty failure.

### Withheld input combinations

This subset consists of evaluation steps with both current inputs > 0.65. It tests a limited distribution shift; past inputs and previous responses remain observable. It is not an unseen-organism or arbitrary compositional-generalization test.

| Setting / family | AM-0 subset nMSE | Quadratic LMS subset nMSE |
|---|---:|---:|
| one_pass / adaptive | 0.03249 | 0.03028 |
| one_pass / resource | 0.01459 | 0.01040 |
| one_pass / delayed | 0.03530 | 0.03532 |
| time_cap / adaptive | 0.01287 | 0.00439 |
| time_cap / resource | 0.00555 | 0.00213 |
| time_cap / delayed | 0.03808 | 0.01438 |

## What the ablations say

The table gives median relative error improvement of AM-0 over the indicated control in the time-capped setting. Positive values favor AM-0. Comparisons of pooling and support formation are limited to this implementation, grammar and tuning grid.

| Family | Over fixed membership | Over no-pool update | Over shuffled outcomes | Over zero output |
|---|---:|---:|---:|---:|
| adaptive | +31.6% | +18.7% | +99.2% | +99.3% |
| resource | -3.7% | +39.5% | +99.5% | +99.4% |
| delayed | +9.2% | +15.3% | +95.6% | +97.0% |

## Resource accounting

Every method has a common 32 KiB persistent-state ceiling. This is **not equal memory use**: the actual sizes are below. The one-pass comparison matches data and updates, not wall time. The time-capped comparison matches a wall-time ceiling and available unique data, not arithmetic counts, replay counts or exact execution time. Shared data generation/history preprocessing is recorded separately. No claim of a fully matched scaling frontier is justified.

| Method | State bytes (range) | One-pass median training seconds | Time-cap update range | Incomplete-data cases |
|---|---:|---:|---:|---:|
| assembly | 6122–6198 | 0.0999 | 5774–11955 | 0 |
| assembly_fixed | 6094–6134 | 0.1012 | 5161–13243 | 0 |
| assembly_no_pool | 6094–6154 | 0.0636 | 13590–21677 | 0 |
| adaptive_lms | 5262–5342 | 0.0321 | 22904–42576 | 0 |
| quadratic_lms | 5235–5307 | 0.0293 | 20964–42333 | 0 |
| history_lms | 5233–5297 | 0.0306 | 25646–43349 | 0 |
| current_lms | 5233–5289 | 0.0313 | 25756–42161 | 0 |
| assembly_shuffled | 6150–6174 | 0.1130 | 8260–13629 | 0 |
| assembly_untrained | 6094–6110 | 0.0001 | 0–0 | 0 |

Maximum observed time-cap overrun: 0.016188 seconds. Deadlines are checked before each update, so a final update can finish after the ceiling. Background scheduling and Python overhead affect these numbers. The untrained control intentionally does no label updates.

Representative AM-0 Python-traced peak allocation: 13,180 bytes, excluding the dataset and interpreter. This is not process peak resident memory or a memory measurement of every model. The state-size accounting includes the full candidate catalogue/scores, rather than counting only active features.

## Correctness and artifacts

- Nine unit tests passed, including positivity/conservation, finite-step limiting, structural replacement, causal encoding, no memory writes during inference, and learning a simple unseen mapping.
- Across 2,000 randomized updates, the independently implemented signed-coordinate learner agreed with the assembly learner to maximum prediction difference 2e-15. This path included 116 births and 114 evictions.
- Maximum component-total discrepancy in that audit: 1.24e-14; smallest component/conformation amount: 0.
- Recomputed metrics from all **354 saved prediction arrays** (84 development, 270 evaluation), verified dataset hashes and source hashes against the recorded Git revisions. The [verifier](../verify_results.py) does not rerun every timed fit; update counts are machine-dependent. Run it with `--development results/development_20261016T062252660199Z --evaluation results/evaluation_20261016T062321995151Z`.
- [Raw evaluation](../results/evaluation_20261016T062321995151Z/measurements.json), [algebra/numerical audit](../results/evaluation_20261016T062321995151Z/audit.json), [metadata](../results/evaluation_20261016T062321995151Z/metadata.json), [summary](../results/evaluation_20261016T062321995151Z/summary.json), and prediction/data NPZ files in the same directory.
- [Development results and selected rates](../results/development_20261016T062252660199Z/development.json). Source before development: `cc2b9c6`; source before evaluation: `2dcdf0d`.
- Timestamps in artifacts are actual UTC timestamps; the user-facing research date is October 15, 2026, America/Los_Angeles. No fabricated dates or retuning from held-out outcomes.

## Limits and next decision

This result rejects a particular static product-assembly abstraction as the sought new learning principle. It does not reject all assembly-based learning. The dynamic support, conserved pools and learning curves should remain available as a negative/control result.

Do not scale AM-0 or spend on GPUs. The next candidate should investigate learned **temporal assembly pathways**: experience changes which transitions are possible and how partial histories are reused. Before coding, specify the transition and credit rule and compare it to weighted automata, grammar learning, reaction networks and recurrent state models. Merely renaming transitions as reactions would repeat the same problem. That investigation can become a separate repository when it has an executable project, with this one linked as the preceding falsification.

No user action or Claude Science access is required for the work completed here. An optional [independent-review prompt](EXTERNAL_REVIEW.md) is available; agreement from another model would not establish novelty. Real biological validation and broader recurrent/chemical/model-bank controls are deferred, not represented by this synthetic pilot.
