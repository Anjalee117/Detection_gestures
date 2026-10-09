# Dataset provenance

The legacy dataset has no signer or session identifiers. Row order and hand shape
cannot establish these identities. Its evaluation uses stratified sample splits;
it does not measure performance on unseen signers or recording sessions.

For newly recorded samples, keep an optional
`model/keypoint_classifier/sample_metadata.csv` with this header:

```csv
sample_sha256,signer_id,session_id
```

Use `training.data.sample_hash(features, label)` to identify each sample; assign
real signer IDs and a new session ID per recording. The hash includes the label
and all 42 normalized float32 features. Metadata must cover every sample exactly
once. Partial records and unknown IDs fail loudly. Do not guess legacy IDs.

When complete metadata exists, the notebook automatically holds out entire
signer/session pairs for validation and testing. Each partition must contain all
26 classes. This holds out sessions, not necessarily whole signers. For stronger
evaluation, collect enough signers to hold out people independently as well.
