# Reproduction guide

[Back to the repository](../README.md) · [Documentation index](README.md)

Run commands from the repository root. The committed CSVs and figures can be inspected without training. The recorded experiment environment is in [ENVIRONMENT.md](ENVIRONMENT.md).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Python 3.11 or newer is required. Training selects MPS, then CUDA, then CPU when available. The context-sweep script currently selects MPS or CPU; the dedicated L4 benchmark uses CUDA.

## Check the mathematics and implementation

These suites need no dataset or checkpoints:

```bash
python -m unittest discover -s tests -p 'test_kernel.py' -v
python -m unittest discover -s tests -p 'test_model.py' -v
python -m unittest discover -s tests -p 'test_ssm.py' -v
```

They cover the convolution/recurrence identity, phase-sign correction, causality, parameter matching, decode-cache size, and diagonal-SSM equivalence. CI runs these same suites on CPU.

## Prepare the dataset

Place the WikiText-2 **raw** splits in this layout:

```text
data/wikitext-2-raw/
├── wiki.train.raw
├── wiki.valid.raw
└── wiki.test.raw
```

Use the raw character corpus, not the pre-tokenized word-level variant. The corpus is not distributed here. The tracked [character vocabulary](../data/char_vocab.json) has 284 symbols and must match the raw training split.

```bash
python -m unittest discover -s tests -v
```

The full suite includes a corpus-dependent test and fails if the raw data is absent; the three suites above are the fresh-clone checks.

## Regenerate the recorded training recipe

These commands train new checkpoints; they do not download existing ones:

```bash
python -m src.train --model resonator --steps 2000 --seed 0
python -m src.train --model transformer --pos_encoding learned --steps 2000 --seed 0
python -m src.train --model transformer --pos_encoding rope --steps 2000 --seed 0
python -m scripts.eval_context_sweep
```

Checkpoints are written under `checkpoints/` and are gitignored. The context sweep requires all three checkpoints and the test corpus. New runs can overwrite result files, so use a separate checkout when preserving the committed evidence. Exact metrics may vary by runtime and hardware.

## Benchmarks and figures

```bash
python -m src.benchmark
python -m src.synthetic_retrieval
python -m scripts.generate_plots
```

The plot generator reads the committed context and memory CSVs; the decay panel is an analytic impulse-envelope calculation. It does not require model checkpoints. The memory chart distinguishes 12 KiB of resonant state from 30 KiB of total decode cache.

For an NVIDIA GPU, `python -m scripts.benchmark_l4` provides the paper-oriented timing protocol. Local MPS timings are evidence about that host; they are not absolute L4 replications.

## Budget and interpretation


Being explicit about the compute budget, because it bounds what these results support:

| Setting | Paper | This Replication |
| :--- | :---: | :---: |
| Optimization steps | 10,000 | **2,000** |
| Seeds | 6 (primary), 3 (breadth) | **1 (seed 0)** |
| Batch size | unstated | 32 |
| Sequence length | 256 | 256 |
| Optimizer | AdamW, lr $5\times10^{-4}$, cosine | identical |
| Hardware | 1x NVIDIA L4 (CUDA) | Apple M5 (MPS) |

**What this budget supports.** The architectural, numerical, and memory findings, all of which are seed-independent and exactly reproducible: the Equation (4) phase-sign error, the diagonal-SSM equivalence, the constant-state-memory result, the half-life decay limit, and the RoPE-versus-learned-PE diagnosis. Re-running `scripts/eval_context_sweep.py` against the trained checkpoints reproduces every digit of the table above, run after run on the same host.

**Note on artifacts.** The checkpoints (116 MB) and the WikiText-2 raw corpus are gitignored, so a fresh clone has neither. Regenerate them with `python -m src.train --model resonator --steps 2000 --seed 0` and the equivalent `--model transformer` runs (`--pos_encoding learned` and `--pos_encoding rope`) after placing `wiki.{train,valid,test}.raw` under `data/wikitext-2-raw/`. Only `data/char_vocab.json` is tracked, and `tests/test_data.py` asserts it still matches what the pipeline rebuilds from the corpus.

**An unexplained discrepancy, stated plainly.** Our absolute numbers come out *better* than the paper's on both models (3.32 vs 3.764 PPL for ResonatorLM, 3.40 vs 4.617 for the Transformer; 64.3% vs 61.31% and 63.6% vs 55.32% accuracy) despite one fifth of the optimization steps. Less training cannot produce better test loss, so our setup and the paper's differ somewhere beyond step count. Two candidates we can name: our character vocabulary is 284 symbols built from the raw training split, against the paper's stated ~256; and at batch 32 the paper's 10,000 steps is 7.6 epochs over WikiText-2 raw where our 2,000 steps is 1.5, so if their batch size was comparable, some of the gap may be overfitting on their side. We have not resolved this, and we flag it rather than presenting our numbers as a match. Every comparison in this repository is therefore *within* our own setup, never across to the paper's absolute values.

**What it does not support.** Any claim about matching the paper's absolute perplexities, or about the size of the ResonatorLM-versus-Transformer quality gap. With one seed we cannot estimate variance, so the 3.322 vs 3.400 gap should be read as a direction, not a measured effect. Reproducing the paper's headline numbers would require the full 10,000-step, six-seed protocol.


[Back to the repository](../README.md)
