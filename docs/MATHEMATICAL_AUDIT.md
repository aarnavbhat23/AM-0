# Mathematical audit: the first candidate fails the strong novelty gate

## Exact representation, not a loose analogy

For fixed active support, define w = b_plus - b_minus and m = b_plus + b_minus. This is invertible:

$$b^+=(m+w)/2,\qquad b^-=(m-w)/2,\qquad m\geq |w|.$$

Free components are determined by c = M - N m. The prediction is exactly w dot phi(z), a degree-two polynomial in the fixed history encoding. The same support, scores and update counter can be retained. Thus the entire learner can be implemented as a polynomial predictor with auxiliary optimizer state m and a bounded feature-admission procedure. This change of variables preserves state size and per-step computational order; it is not a universal-computation argument that would trivialize all architectures.

## Exact learning update

Let h = eta clip(y - w dot phi, -2, 2) phi for active coordinates, sigma = sign(h), and d = min(abs(h), 1). Write the concentration opposite the desired change as

$$o=(m-\sigma w)/2.$$

Dissolution gives

$$w^*=w+\sigma d o,\qquad m^*=m-d o,\qquad c^*=M-Nm^*.$$

With the same availability g and shared limiter alpha defined in the rule, growth gives

$$w'=w^*+\sigma\alpha|h|g,\qquad m'=m^*+\alpha|h|g.$$

The zero-h case changes nothing. When abs(h) <= 1 coordinatewise, this simplifies to

$$w'=w+h(m/2+\alpha g)-|h|w/2.$$

The first term after w is a diagonal, state-dependent error update; the final term is an error-dependent shrinkage. The clipped residual is the negative derivative of a Huber loss with threshold 2. The score trace is therefore a smoothed negative coordinate gradient in the polynomial feature space. This **is not a proof that AM-0 is literally standard SGD, exponentiated gradient, natural gradient, or a specific published optimizer**. The dependence of g and alpha on the substrate matters. It is an exact alternative implementation as a conventional polynomial hypothesis class with stateful constrained coefficient updates.

For a step without structural change, both dissolution and growth move each w in the sign of e*phi. Therefore e*Delta(y_hat) >= 0 on that training example. This local direction result does not prevent overshooting, imply loss decrease for finite eta, imply generalization, or survive arbitrary eviction. No convergence theorem is claimed.

## Learned assembly membership

The catalogue contains every degree-one and degree-two feature allowed by the grammar. Admitting a new assembly adds a feature; dissolving an assembly deletes its coefficient and releases its resource charges. It does **not** discover a new computational operator. The correlation scores use all candidate features and the shared outcome residual. This has direct overlap with adaptive polynomial feature construction [R2, R3]. Our heuristic is not claimed to reproduce those papers' algorithms or guarantees.

The learned membership is real data-dependent structure, but structural change alone does not establish a new learning principle. Nor does naming a coefficient a conformation. Computational primitives do not become new simply because a physical realization is different.

## What conservation adds

For any realizable state,

$$N|w|\leq Nm\leq M.$$

Thus the finite component supply imposes overlapping weighted L1 constraints on polynomial coefficients; opposing conformations can make the bound strictly tighter. It also couples learning rates across features competing for components. This could be a useful regularizer or a source of interference. It is a testable design choice, not evidence for a new intelligence class. The no-pool ablation tests its predictive contribution.

## Limits and nearby methods

| Limit or design choice | Reduction / relationship | Status |
|---|---|---|
| One free/sequestered component | Bounded gate | Already established in the earlier survey |
| Fixed assembly catalogue, fixed readout | Polynomial predictor with (w,m) optimizer state | Exact, derived above and checked by an independent implementation |
| Score-based assembly birth | Adaptive interaction-feature selection | Direct prior-art overlap; not exact equality to one named algorithm |
| Separate pools routed by observed context | Stored model bank with capacity constraints | Architectural interpretation; not implemented here |
| Copy a chosen assembly to another pool | Copy selected coefficients/support with resource accounting | Does not itself provide a new transfer-learning rule |
| Multiple decay rates on memory | Multiscale memory | Known broad principle; adding timescales alone is insufficient |
| Chemical implementation of learning | Chemical associative/generative learners | Existing primary work [R4, R5]; chemistry alone is not novelty |

## Decision before performance results

**Do not promote AM-0 as the next neural network, or as a new learning primitive.** It is a fully specified, non-neural diagnostic learner whose prediction representation and credit signal are familiar. Its particular conservation-coupled update may be a variant worth studying, but we have not established that the exact variant is new or useful.

Run a small diagnostic to verify that it learns something, preserves its invariants, and has the predicted equivalence. Do not launch the broader confirmatory study or a scaling run on the strength of a low synthetic error. A result favoring the assembly implementation would support a constrained-feature-learning tradeoff, not rescue the stronger novelty claim.

This rejects **AM-0**, not every possible assembly-memory architecture. The next serious candidate would need experience to alter the dynamics that perform computation (for example reusable transition pathways), with an explicit credit rule and a nontrivial cost/learning benefit. That overlaps the second survey direction, temporal assembly pathways, and requires a separate audit. Do not turn that suggestion into an unsupported novelty claim.

References and access limits are in [SOURCES.md](SOURCES.md).
