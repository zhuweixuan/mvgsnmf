# Data provenance and terms

We derived the matrices in `data/` from prescription data distributed with the
PTM project:

- Repository: <https://github.com/yao8839836/PTM>
- PTM revision used to derive these matrices:
  `597cf09fece58c787093212c4063c3ca9aeea633`
- Related work: Yao et al. (2018), the PTM traditional Chinese medicine
  prescription topic-model project.

The PTM README identifies CKCEST as the copyright holder for the source data
and states that the data are for research use only, excluding commercial use,
sale, and monetization.

To support reproducibility, we include only the derived inputs used in our
reported experiments. The prescription-level matrices contain sequential
internal identifiers and encoded herb and symptom indicators; they do not
contain raw prescription text or direct personal identifiers. We do not include
raw `prescriptions.txt`, filtered or mapped free text, or parsing logs.

The repository's MIT License applies to the source code and documentation. The
derived datasets and matrices under `data/` follow the upstream source terms.
See [`data/README.md`](data/README.md) for file descriptions and
[`data/SHA256SUMS`](data/SHA256SUMS) for checksums.
