# Assembly memory

Can persistent, resource-limited assemblies provide a useful learning operation?

This repository isolates one candidate from [Nature Compute Lab](https://github.com/aarnavbhat23/nature-compute-lab). The first candidate, AM-0, is completely specified but **already fails the strong novelty gate**: its assembly state can be rewritten exactly as polynomial coefficients with auxiliary optimizer state and adaptive feature selection. We are keeping a small falsification prototype to test the dynamics, not calling it a new intelligence or an ASI architecture.

The first diagnostic is complete: [results and decision](docs/RESULTS.md). Nine tests passed; all 354 saved development/evaluation prediction arrays were checked. Under the common time ceiling, quadratic LMS beats AM-0 on all 15 held-out systems. AM-0 is not being promoted to a scaling project.

Start with the [learning rule](docs/LEARNING_RULE.md), [mathematical audit](docs/MATHEMATICAL_AUDIT.md), and [diagnostic protocol](docs/PROTOCOL.md). The [source audit](docs/SOURCES.md) distinguishes natural observations from software assumptions. [External review](docs/EXTERNAL_REVIEW.md) includes an optional scientific-review prompt; no external review is required or claimed.

The initial repository is private. It contains no biological sequences, private experimental data, credentials or wet-lab protocols. No open-source license has been selected. Keep real timestamps and negative results. Conclusions require independent scrutiny.

## Reproduce

Python 3.11 or newer, NumPy as pinned in `requirements.txt`. ```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run.py --phase development
# Supply the selected_rates.json written by the development run:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run.py --phase evaluation --selected results/DEVELOPMENT_RUN/selected_rates.json
```

Source and protocol must be committed before running. Each run writes a new timestamped directory with its revision, hashes, data, predictions and measurements. Existing results are never overwritten. Time-capped update counts vary with the machine. The [implementation notes](docs/IMPLEMENTATION_LOG.md) record the pre-experiment numerical correction.
