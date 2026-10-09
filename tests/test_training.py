import csv

import numpy as np
import pytest

from training.data import load_dataset, load_groups, sample_hash, split_indices


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


@pytest.mark.parametrize("row", [[0] * 85, [0] * 42, [0] + [float("nan")] * 42,
                                   [26] + [0] * 42, [0.5] + [0] * 42,
                                   [0] + [2] * 42, [0] + ["oops"] * 42])
def test_malformed_dataset_is_rejected(tmp_path, row):
    path = tmp_path / "bad.csv"
    write_rows(path, [row])
    with pytest.raises(ValueError):
        load_dataset(path)


def test_independent_stratified_and_group_splits():
    labels = np.tile(np.arange(26), 20)
    groups = np.repeat(np.arange(20), 26)
    for metadata in (None, groups):
        train, val, test = split_indices(labels, metadata)
        assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
        assert set(np.r_[train, val, test]) == set(range(len(labels)))
        assert all(set(labels[part]) == set(range(26)) for part in (train, val, test))
        if metadata is not None:
            assert not set(groups[train]) & set(groups[val])
            assert not set(groups[train]) & set(groups[test])
            assert not set(groups[val]) & set(groups[test])


def test_metadata_requires_complete_genuine_ids(tmp_path):
    x = np.zeros((2, 42)); x[1, 2] = 0.5
    y = np.array([0, 1])
    path = tmp_path / "metadata.csv"
    assert load_groups(path, x, y) is None
    rows = [[sample_hash(x[i], y[i]), "signer-a", f"session-{i}"] for i in range(2)]
    write_rows(path, [["sample_sha256", "signer_id", "session_id"], *rows])
    assert len(set(load_groups(path, x, y))) == 2
    rows[0][1] = "unknown"
    write_rows(path, [["sample_sha256", "signer_id", "session_id"], *rows])
    with pytest.raises(ValueError, match="genuine"):
        load_groups(path, x, y)
    write_rows(path, [["sample_sha256", "signer_id", "session_id"], rows[1]])
    with pytest.raises(ValueError, match="exactly one"):
        load_groups(path, x, y)
