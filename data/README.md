# Derived model inputs

The minimum active inputs for `config/paper_full.yaml` are:

- `matrix_from_mappings_herb_presence.csv`: prescription-by-herb binary matrix.
- `matrix_from_mappings_symptom.csv`: prescription-by-symptom binary matrix.
- `herb_category_onehot.csv`: herb category features.
- `herb_feature_matrix_793.csv`: herb property features.
- `symptom_feature_matrix.csv`: symptom property features.
- `herb_cooccurrence_pairs.csv`: herb co-occurrence index pairs.
- `symptom_cooccurrence_pairs.csv`: symptom co-occurrence index pairs.
- `herb_mutual_exclusion_pairs_index_pairs.csv`: herb incompatibility pairs.
- `symptom_herb_tcm_mesh.txt`: symptom-herb knowledge links.
- `herb_alias_user.csv`: herb alias mapping used by the knowledge loader.

The prescription matrices contain 33,729 rows, 793 herb columns, and 390
symptom columns. The first column is the internal `prescription_id`.

The dosage matrix is intentionally absent: the manuscript experiments did not
activate the dosage reconstruction loss. See `../DATA_TERMS.md` before sharing
these derived files.
