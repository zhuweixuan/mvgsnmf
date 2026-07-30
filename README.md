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

## Decremental ablation chain

The manuscript evaluates six configurations by starting from full MV-GSNMF
and cumulatively removing one add-on at a time. `Graph` means bilateral herb
and symptom graph regularization. `Know` is TCM MeSH knowledge coupling, and
`Pair` is the herb-pair co-occurrence view.

| Stage | Manuscript variant | Active add-ons | Cumulative change |
|---:|---|---|---|
| 00 | Full MV-GSNMF | Graph + Know + Pair + L1 | Full objective |
| 01 | SL-CNMF + Graph + Pair + L1 | Graph + Pair + L1 | Remove `know_hs` |
| 02 | SL-CNMF + Graph + L1 | Graph + L1 | Then remove `pair` |
| 03 | SL-CNMF + Graph | Graph | Then remove `l1` |
| 04 | SL-CNMF + HerbGraph | Herb graph only | Then remove `graph_s` |
| 05 | Reconstruction-only SL-CNMF | None | Then remove `graph_h` |

The herb-presence (`ph`) and symptom-presence (`ps`) reconstruction terms and
nonnegativity remain active in every stage. The dosage view (`pd`) is disabled
because dosage-aware modeling is outside the scope of the present study.
Consequently, stage 05 is a reconstruction-only, shared-prescription-factor
two-view NMF, not the separate concatenated-data `Vanilla NMF` baseline.

### Full paper sweep: all six variants

Run the following command from the project root. It explicitly reproduces the
paper design: six stages, five seeds, and eight topic counts, for 240 runs.
The runner creates the entire chain in one invocation; separate commands are
not required for individual stages.

PowerShell:

```powershell
python scripts/run_ablation_multiseed.py `
  --base_config config/best_v4.yaml `
  --seeds 42 43 44 45 46 `
  --k_values 5 10 15 20 25 30 35 40 `
  --drop_order know_hs pair l1 graph_s graph_h `
  --pretrain_iters 100 `
  --output_root artifacts/ablation_multiseed `
  --resume
```

Bash:

```bash
python scripts/run_ablation_multiseed.py \
  --base_config config/best_v4.yaml \
  --seeds 42 43 44 45 46 \
  --k_values 5 10 15 20 25 30 35 40 \
  --drop_order know_hs pair l1 graph_s graph_h \
  --pretrain_iters 100 \
  --output_root artifacts/ablation_multiseed \
  --resume
```

The base configuration uses `device: gpu`; these batch runners inherit the
device from the YAML file. `--resume` skips only runs that already contain
`summary.json`, `factors.npz`, `metrics.json`, and `config.yaml`.

The generated stage directories are:

```text
00_active_graph_h__graph_s__l1__pair__know_hs/
01_active_graph_h__graph_s__l1__pair/
02_active_graph_h__graph_s__l1/
03_active_graph_h__graph_s/
04_active_graph_h/
05_recon_only_ph_ps_nonneg/
```

Each stage stores results as
`<stage>/seed_<seed>/K_<K>/{summary.json,factors.npz,metrics.json,config.yaml}`;
the output root also contains `manifest.json`.

### Short chain check

This command runs all six stages at only `seed=42, K=30`. It is useful for
checking the pipeline but is not a replacement for the 240-run paper sweep.

```powershell
python scripts/run_ablation_multiseed.py `
  --base_config config/best_v4.yaml `
  --seeds 42 `
  --k_values 30 `
  --drop_order know_hs pair l1 graph_s graph_h `
  --pretrain_iters 100 `
  --output_root artifacts/ablation_chain_seed42_K30 `
  --resume
```

## NMF comparator family

The separate baseline runner evaluates the five NMF-side comparison variants:
reconstruction-only SL-CNMF, concatenated-data Vanilla NMF, independent NMF
with post-hoc Procrustes alignment, sparse NMF, and GNMF. These are comparison
models and are not additional steps in the six-stage decremental chain.

```powershell
python scripts/run_ppl_multiseed_baselines.py `
  --base_config config/best_v4.yaml `
  --seeds 42 43 44 45 46 `
  --k_values 5 10 15 20 25 30 35 40 `
  --eval_split test `
  --output_root artifacts/ppl_multiseed_compare `
  --auto_resume
```

## Other experiment entry points

```bash
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
included. Dosage-aware prescription modeling was not evaluated in this study
and is one of our planned directions for future work. Readers are welcome to
follow this repository for future updates.

No dosage matrix is distributed in this repository. Every released experiment
sets `loss_switches.pd: false`; in this mode the loader neither validates nor
reads the reserved `files.herb_dosage` path. It instead creates a
shape-compatible, all-zero sparse placeholder, so training and the smoke test
run normally while the dosage CSV is absent. The non-public file path is also
listed in `.gitignore` to prevent accidental publication. Enabling `pd` in
future work will require adding a local dosage dataset that is not part of this
release.

The included matrices contain sequential prescription identifiers and encoded
herb/symptom indicators, not raw prescription text. They remain derived from a
third-party research dataset and are not covered by the code terms. Read
[`DATA_TERMS.md`](DATA_TERMS.md) before redistribution and verify every input
with [`data/SHA256SUMS`](data/SHA256SUMS).

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
