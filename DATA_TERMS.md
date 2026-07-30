# Data provenance and redistribution notice

The prescription source used to derive the matrices in `data/` was distributed
with the PTM project:

- Repository: <https://github.com/yao8839836/PTM>
- Local source revision used during this work:
  `597cf09fece58c787093212c4063c3ca9aeea633`
- Related work: Yao et al. (2018), the PTM traditional Chinese medicine
  prescription topic-model project.

The PTM README attributes the source data copyright to CKCEST and limits the
data to research use, excluding commercial use, sale, or monetization. It does
not provide a standard data license or an explicit general redistribution
grant.

Therefore:

1. Raw `prescriptions.txt`, filtered prescription text, mapped free text, and
   parsing logs are not included in this package.
2. The included row-level binary matrices are derived data, but they may still
   be subject to the source restrictions. They are provided here only for
   private manuscript review and research reproducibility.
3. Do not publish or redistribute the `data/` directory until the authors have
   confirmed that the source terms permit it or have obtained permission.
4. The repository's MIT License for code does not apply to the files under
   `data/`.
5. The matrices use sequential internal prescription identifiers; no raw text,
   phone number, email address, or government identifier is included.

For a public repository without confirmed redistribution permission, omit the
two prescription-level matrices and distribute only this notice, schemas,
checksums, and instructions for authorized users to regenerate or place the
files locally.
