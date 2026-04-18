# Model archive

Inventory of every nanoGPT write-filter checkpoint Jay has trained,
with reproducibility notes. Ckpts live outside this repo in the
nanoGPT fork (`~/Desktop/Personal/Projects/nanoGPT/`), which is a
private fork; they are NOT tracked in git. This file documents where
each ckpt is, what dataset produced it, and how to regenerate if the
directory is lost.

## Inventory (as of 2026-04-18)

| Alias | Path | Config | Dataset dir | Size | Timestamp |
|-------|------|--------|-------------|-----:|-----------|
| v3 | `out-merken-v3/ckpt.pt` | `config/train_merken_v3.py` | `data/merken_v3/` | 10.0 MB | 2026-04-16 19:55 |
| v4 | `out-merken-bpe/ckpt.pt` | `config/train_merken_bpe.py` | `data/merken_bpe/` | 10.6 MB | 2026-04-16 20:25 |
| v5 | `out-merken-bpe-v5/ckpt.pt` | `config/train_merken_bpe_v5.py` | `data/merken_bpe_v5/` | 10.6 MB | 2026-04-17 08:59 |
| v6 | `out-merken-bpe-v6/ckpt.pt` | `config/train_merken_bpe_v6.py` | `data/merken_bpe_v6/` | 10.6 MB | 2026-04-17 12:20 |
| **v7** | **`out-merken-bpe-v7/ckpt.pt`** | **`config/train_merken_bpe_v7.py`** | **`data/merken_bpe_v7/`** | **10.8 MB** | **2026-04-17 20:04 (graduated)** |
| v8 | `out-merken-bpe-v8/ckpt.pt` | `config/train_merken_bpe_v8.py` | `data/merken_bpe_v8/` | 10.8 MB | 2026-04-17 20:36 (archived) |
| v9 | `out-merken-bpe-v9/ckpt.pt` | `config/train_merken_bpe_v9.py` | `data/merken_bpe_v9/` | 10.8 MB | 2026-04-18 10:45 (ablation: v7 recipe minus knowledge_update; REJECTED as replacement, see HYPOTHESES `H_v9_pragmatic`) |

Pre-v3 checkpoints (`out-merken/`, `out-merken-bpe/` pre-rename) exist
on disk but are superseded and not documented here.

## Architecture reference

All BPE checkpoints (v4 onward) share the same architecture:

| Param | Value |
|-------|------:|
| n_layer | 4 |
| n_head | 4 |
| n_embd | 128 |
| vocab_size | 512 |
| dropout | 0.1 |
| approx params | ~800K |

Differences across versions are:
- `block_size`: 128 for v4/v5/v6, **256 for v7/v8**.
- `max_iters` and `always_save_checkpoint`: see per-config files.
- Training dataset composition (see per-prepare.py files).

v3 is char-level (81-token vocab); v4 onward uses BPE with explicit
`DECISION`/`NOISE` single tokens.

## How to load a specific ckpt from merken

`merken.classifiers.nanogpt.NanoGPTWriteDecider` auto-detects char vs
BPE from `meta.pkl`. To swap which version shadow mode uses:

    export MERKEN_SHADOW=nanogpt
    export MERKEN_SHADOW_NANOGPT_CKPT=.../out-merken-bpe-v6/ckpt.pt
    export MERKEN_SHADOW_NANOGPT_META=.../data/merken_bpe_v6/meta.pkl

v7 is the default (set in `~/.claude/hooks/merken-save.sh` and in
the Claude Desktop MCP config).

## How to regenerate a lost ckpt

Each version's prepare.py + config tuple is self-contained and
reproducible given the same source data:

    cd ~/Desktop/Personal/Projects/nanoGPT
    python data/merken_bpe_vN/prepare.py       # rebuilds train.bin/val.bin/meta.pkl/tokenizer.json
    python train.py config/train_merken_bpe_vN.py

External inputs needed (live in the engram repo or local caches):
- `engram/experiments/loop_quality/scenarios/*.json` (always)
- `/tmp/organic_train.json` (v5 onward -- regenerate via
  `python -m experiments.consolidation.extract_organic_training_data`)
- `engram/experiments/data/markdown_noise_v6.json` (v6 onward --
  regenerate via `experiments.consolidation.generate_markdown_noise`)
- `engram/data/merken_labels_v7.jsonl` (v7 onward -- regenerate via
  PR #12 bootstrap pipeline; gitignored)

Randomness: all prepare.py scripts seed `random.seed(2026)`.
Tokenizer training is deterministic given the same input.

## Known facts per version

| version | markdown FPR | jay_vstash | organic_val | knowledge_update DEC | real oracle accuracy |
|---------|-------------:|-----------:|------------:|---------------------:|---------------------:|
| v4 | 100.0% | 95% | 100% | 100% | (not measured) |
| v5 | 100.0% | 100% | 100% | 100% | (not measured) |
| v6 | 66.7% | 100% | 100% | 99.3% | 84.7% (skip-set only) |
| **v7** | **0.0%** | **100%** | **100%** | 97.4% | **80.9%** (population-weighted) |
| v8 | 100.0% | 100% | 100% | 99.2% | (not measured) |

See `experiments/nanogpt/CONFUSION_MATRIX.md` for the v7 full confusion
matrix + calibration analysis, and `experiments/nanogpt/RESULTS.md`
for the per-version change-history narrative.

## Rollback procedure

To revert `MERKEN_SHADOW` to a previous version (e.g. if v7 misbehaves
in production):

1. Update `~/.claude/hooks/merken-save.sh` -- change the
   `MERKEN_SHADOW_NANOGPT_CKPT` / `_META` env vars to the old version.
2. Same change in Claude Desktop MCP config
   (`~/Library/Application Support/Claude/claude_desktop_config.json`).
3. New sessions will shadow with the old version; previous audit rows
   remain tagged with whichever version was active when written.

No database migration required. Audit rows record the `policy`
string of the decider, not an implicit version -- you can always tell
which model generated a given row by parsing the reason field.
