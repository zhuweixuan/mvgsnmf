# MV-GSNMF reproducibility package

This directory contains the model implementation, the configurations used in
the manuscript, the minimum derived inputs needed by the reported model, and
compact aggregate results. It intentionally excludes checkpoints, per-seed
training directories, logs, caches, virtual environments, and raw prescription
text.

## Create the environment

The reported GPU rerun was verified with Python 3.13.7, NumPy 2.3.1,
SciPy 1.17.1, pandas 2.3.2, scikit-learn 1.8.0, PyYAML 6.0.3, tqdm 4.67.3,
CuPy 14.1.1, CUDA runtime 12.9, and an NVIDIA RTX 4060 Ti.

For the GPU environment:

```bash
conda env create -f environment.yml
conda activate mvgsnmf
python -c "import cupy as cp; print(cp.__version__, cp.cuda.runtime.runtimeGetVersion(), cp.cuda.Device())"
```

On Windows, NVRTC can fail when its temporary path contains Chinese or other
non-ASCII characters. The backend therefore creates a dedicated subdirectory
under the system temp directory. If that system path is not ASCII-only, set an
explicit writable location before running:

```powershell
$env:MVGSNMF_CUDA_TMP = "C:\mvgsnmf_cuda_tmp"
```

This directory holds only regenerable NVRTC temporary files and CuPy kernels.

For a CPU-only environment:

```bash
conda create -n mvgsnmf-cpu python=3.13 pip -y
conda activate mvgsnmf-cpu
python -m pip install -r requirements.txt
```

Run a short end-to-end check from the project root:

```bash
python main.py --smoke --device gpu
```

Use `--device cpu` for the CPU path. The smoke test uses 128 prescriptions,
one pretraining iteration, and two main iterations; it is not a scientific
experiment.

## Reproduce the main training profile

`config/paper_full.yaml` is the full profile used by the final experiments.
All paths are relative to this project.

PowerShell:

```powershell
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python main.py --config config/paper_full.yaml --device gpu
```

Bash:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --config config/paper_full.yaml --device gpu
```

Full training creates a timestamped directory under `artifacts/`. It is much
longer than the smoke test. Small numerical differences can occur across GPU,
CUDA, and sparse-library versions.

`config/best_v4.yaml` is the reconstruction-only base configuration used by
the sweep scripts, which explicitly construct their own experimental loss
profiles.

## Experiment entry points

The main paper-facing scripts are:

```bash
python scripts/run_ablation_multiseed.py --base_config config/best_v4.yaml
python scripts/run_ppl_multiseed_baselines.py --base_config config/best_v4.yaml
python scripts/revision/run_hyperparam_sensitivity.py --base_config config/best_v4.yaml
python scripts/revision/run_structure_sensitivity.py --base_config config/best_v4.yaml --device gpu
python scripts/revision/run_tfidf_ablation.py --base_config config/best_v4.yaml
python scripts/revision/run_wilcoxon_significance.py
```

These batch commands can generate many runs. Their heavy outputs are excluded
from this package; only aggregate tables, statistics, and figures are retained
under `paper_results/`.

## Data scope

The model starts from versioned, derived matrices in `data/`. Raw PTM/CKCEST
prescription text, filtered free text, and intermediate parsing logs are not
included. The reported dosage view was disabled in every paper experiment
(`loss_switches.pd: false`), so the 230 MiB dosage CSV is also omitted. The
loader creates a shape-compatible zero placeholder when that view is disabled;
enabling `pd` still requires the actual dosage file.

The included matrices contain sequential prescription identifiers and encoded
herb/symptom indicators, not raw prescription text. They remain derived from a
third-party research dataset and are not covered by the code terms. Read
[`DATA_TERMS.md`](DATA_TERMS.md) before redistribution and verify every input
with [`data/SHA256SUMS`](data/SHA256SUMS).

The deterministic extraction scripts are maintained outside the original
MV-GSNMF model directory and depend on source files whose redistribution rights
are unresolved. Consequently, this package documents the provenance and begins
from the frozen derived matrices; it does not claim a from-scratch rebuild from
the restricted raw corpus.

## Directory layout

```text
config/          paper and sweep-base YAML configurations
data/            minimum derived inputs and checksums
gsnmf/           model, loading, training, inference, and evaluation code
paper_results/   compact aggregate manuscript results
scripts/         reproducibility and analysis entry points
artifacts/       generated locally; ignored by version control
```

See [`RELEASE_CONTENTS.md`](RELEASE_CONTENTS.md) for the exact inclusion and
exclusion policy.

## Licensing

No open-source license was present in the working project, so this package does
not assign one automatically. The current code notice is in [`LICENSE`](LICENSE).
Replace it with the authors' chosen code license before an open-source release.
The data restrictions in [`DATA_TERMS.md`](DATA_TERMS.md) remain separate.
