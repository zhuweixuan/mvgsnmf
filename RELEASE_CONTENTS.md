# Repository contents

## Included files

- The `gsnmf` Python package and the main training entry point.
- YAML profiles for the full model and experiment sweeps.
- Environment definitions and pinned core/GPU dependencies.
- A full-file checksum list in `MANIFEST-SHA256.txt`.
- The derived matrices, graphs, features, aliases, and knowledge links used by
  the reported experiments.
- Aggregate ablation, baseline, sensitivity, significance, and unified results.
- Reproduction and analysis scripts.

## Not included

- Model checkpoints and per-run factor files.
- Per-seed, per-K, and per-value experiment directories.
- Training logs, caches, bytecode, IDE files, and virtual environments.
- Raw PTM prescription text, filtered or mapped free text, and parsing logs.
- Dosage data. Dosage reconstruction is disabled in the reported experiments,
  and dosage-aware modeling is planned future work.

We keep the repository focused by providing compact aggregate results and
excluding regenerable run artifacts.
