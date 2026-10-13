# ResonatorLM Audit

**Independent replication, numerical verification, and baseline analysis of a resonant language model.**

A from-scratch PyTorch implementation of [ResonatorLM (arXiv:2607.05583v2)](https://arxiv.org/abs/2607.05583v2), with a parameter-matched Transformer, a RoPE control, and executable checks of the underlying mathematics.

[Findings](docs/FINDINGS.md) · [Reproduce](docs/REPRODUCTION_GUIDE.md) · [Code](src/) · [Reports](docs/README.md) · [Raw results](results/README.md)

> **Scope:** ~6M parameters, WikiText-2 raw characters, 2,000 training steps, seed 0. This is a reduced-budget audit, not a full reproduction of the paper's 10,000-step, six-seed study. Checkpoints and corpus files are not included.

## The result in one figure

![Two panels comparing test perplexity and next-character accuracy across 256, 512, and 1024 characters. ResonatorLM stays near 3.3 perplexity; the learned-position Transformer degrades to 20.0 and the RoPE control to 5.59. All models were trained at length 256 with seed 0.](results/figures/context_sweep.png)

**The choice of positional encoding changes the comparison.** In this audit's zero-shot context sweep, RoPE reduces Transformer perplexity at length 1024 from 20.001 to 5.588. ResonatorLM remains at 3.293. These are within-setup comparisons from one seed; they do not estimate variance or establish a general quality ranking.

| Model | PPL at 256 | PPL at 512 | PPL at 1024 |
|:--|--:|--:|--:|
| ResonatorLM | 3.322 | 3.301 | 3.293 |
| Transformer · learned positions | 3.401 | 11.169 | 20.001 |
| Transformer · RoPE | 3.210 | 3.536 | 5.588 |

Lower is better. All models trained at length 256; 512 and 1024 measure zero-shot extrapolation. Values are rounded directly from [context_sweep_results.csv](results/context_sweep_results.csv). Primary training metrics use a separate evaluation pass; see the [results index](results/README.md).

## What the audit examines

| Finding | Evidence to inspect | Boundary |
|:--|:--|:--|
| A phase-sign correction makes convolution and recurrent execution agree | [Kernel tests](tests/test_kernel.py) · [Derivation](docs/FINDINGS.md) | Numerical correctness of the mixer |
| The resonant operator has a diagonal complex SSM formulation | [Equivalent operator](src/ssm.py) · [Equivalence tests](tests/test_ssm.py) | Operator equivalence, not an empirical comparison with every SSM architecture |
| Decode cache stays constant in sequence length | [Cache-size test](tests/test_model.py) | Batch 1, float32: 12 KiB resonant state + 18 KiB local buffers = **30 KiB** across six layers |
| Positional encoding changes extrapolation behavior | [Transformer implementation](src/transformer.py) · [Context sweep](results/context_sweep_results.csv) | Learned positions are clamped beyond the training horizon in this implementation |
| Damping limits each mode's impulse retention | [Decay audit](src/synthetic_retrieval.py) · [Recorded decay](results/half_life_decay_audit.csv) | Impulse decay is distinct from a measured long-range retrieval benchmark |

The [full findings](docs/FINDINGS.md) and [reproduction report](docs/REPRODUCTION_REPORT.md) contain the original analysis and qualifications. Local timing results come from Apple M5/MPS and should not be read as NVIDIA L4 speed reproductions.

## Choose a reading path

**For researchers:** start with the [findings](docs/FINDINGS.md), then inspect the [paper specification](docs/PAPER_SPEC.md), [implementation assumptions](docs/ASSUMPTIONS.md), and [reproduction guide](docs/REPRODUCTION_GUIDE.md). The [results index](results/README.md) maps each artifact to its producing script.

**For recruiters and engineering reviewers:** the most useful code path is [mixer](src/resonator.py) → [language model](src/resonator_lm.py) → [training](src/train.py). Follow it with the [numerical tests](tests/test_kernel.py) and [RoPE control](src/transformer.py). Together these show architecture implementation, experiment design, numerical debugging, and evidence-aware reporting through concrete artifacts.

## Run the data-free checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m unittest discover -s tests -p 'test_kernel.py' -v
python -m unittest discover -s tests -p 'test_model.py' -v
python -m unittest discover -s tests -p 'test_ssm.py' -v
```

Requires Python 3.11+. These suites need no corpus or checkpoints. For data preparation, training, context evaluation, and benchmarks, follow the [reproduction guide](docs/REPRODUCTION_GUIDE.md).

## Repository map

```text
src/         Model implementations, training, evaluation, benchmarks
scripts/     Experiment orchestration, context sweeps, figure generation
tests/       Numerical identities, model behavior, data consistency
docs/        Research reports, specifications, reproduction guide
results/     Recorded CSV/JSONL evidence and generated figures
data/        Tracked character vocabulary; raw corpus is gitignored
```

Maintained by [Kanishk Paul](https://github.com/kanishkpaul). Independent of the original paper's authors. Code is released under the [MIT license](LICENSE). To cite this software, use [CITATION.cff](CITATION.cff).
