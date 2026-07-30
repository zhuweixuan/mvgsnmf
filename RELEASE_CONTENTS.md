# Release contents

## Included

- The `gsnmf` Python package and the main training entry point.
- Relative-path YAML profiles for the full paper model and the sweep base.
- Environment creation and pinned core/GPU dependency files.
- A full-file checksum list in `MANIFEST-SHA256.txt`.
- Minimum derived matrices and small graph, feature, alias, and knowledge files.
- Aggregate ablation, baseline, sensitivity, significance, and unified results.
- Principal reproduction and analysis scripts.

## Intentionally excluded

- All 641 `factors.npz` files and other checkpoints.
- Per-seed/per-K/per-value experiment directories.
- Training logs, caches, bytecode, IDE files, and virtual environments.
- Raw PTM prescription text, filtered or mapped free text, and parsing logs.
- The 230 MiB dosage matrix, because every reported experiment has
  `loss_switches.pd: false`.
- Historical repair scripts, one-off print helpers, and obsolete comparison
  scripts containing machine-specific paths.

The source working directory was approximately 1.70 GiB, almost entirely due
to experiment artifacts. This package retains compact aggregate evidence rather
than those regenerable outputs.
