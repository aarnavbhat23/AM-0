# Implementation notes before the diagnostic

The equation/protocol commit precedes executable tests and model measurements. During numerical unit testing, the deliberately extreme eta=10 stress case exposed a floating-point issue in the shared growth limiter: an incrementally tracked free pool could retain about 1e-42 units while the coefficient/occupancy representation rounded the same pool to zero. Even that negligible demand could restrict growth in every other pool because the limiter uses a ratio.

Before any development or held-out fit, we specified a numerical depletion floor: a free fraction <= 1e-12 has zero **growth availability** in both implementations. The amount is not deleted or replenished, so component accounting remains intact. This is a numerical convention, not a biologically measured threshold. The ordinary and extreme-step equivalence tests must pass with this rule. The original failure was an implementation/numerical validation result; no held-out performance was used to choose the correction.

The executable rule is otherwise the written AM-0 rule. Full source hashes and the precise revision are stored per run. The simulator equations and controller are engineering choices, not inferred from a biological dataset.

New repository settings: private visibility, vulnerability alerts enabled, GitHub Actions default token permissions set to read, Actions cannot approve pull-request reviews. There is no workflow with write access or external credential. No open-source license has been selected.
