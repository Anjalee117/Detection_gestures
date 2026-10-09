"""Strict loading of the 26-class, 42-feature landmark dataset."""
import csv
import hashlib
from pathlib import Path

import numpy as np


def load_dataset(path):
    records = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if len(row) != 43:
                raise ValueError(f"{path}:{line}: expected 43 columns, got {len(row)}")
            try:
                values = np.asarray(row, dtype=np.float32)
            except ValueError as error:
                raise ValueError(f"{path}:{line}: non-numeric value") from error
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line}: non-finite value")
            if values[0] != int(values[0]) or not 0 <= values[0] < 26:
                raise ValueError(f"{path}:{line}: label must be an integer in [0, 25]")
            if np.max(np.abs(values[1:])) > 1.00001:
                raise ValueError(f"{path}:{line}: landmarks must be normalized to [-1, 1]")
            records.append(values)
    if not records:
        raise ValueError(f"{path}: empty dataset")
    data = np.asarray(records, dtype=np.float32)
    if set(data[:, 0].astype(int)) != set(range(26)):
        raise ValueError(f"{path}: dataset must contain all 26 classes")
    return data[:, 1:], data[:, 0].astype(np.int32)


def sample_hash(features, label):
    values = np.r_[np.float32(label), np.asarray(features, dtype=np.float32)]
    return hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()


def load_groups(path, features, labels):
    """Optional provenance; never infer people or sessions from row order."""
    if not Path(path).exists():
        return None
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_sha256", "signer_id", "session_id"]:
            raise ValueError("Metadata requires sample_sha256,signer_id,session_id columns")
        metadata = list(reader)
    expected = [sample_hash(x, y) for x, y in zip(features, labels)]
    indexed = {row["sample_sha256"]: row for row in metadata}
    if len(indexed) != len(metadata) or set(indexed) != set(expected):
        raise ValueError("Metadata must contain exactly one record for every dataset sample")
    groups = []
    for key in expected:
        row = indexed[key]
        if any(not row[name].strip() or row[name].strip().lower() == "unknown"
               for name in ("signer_id", "session_id")):
            raise ValueError("Session evaluation requires genuine signer and session IDs")
        groups.append((row["signer_id"].strip(), row["session_id"].strip()))
    # Tuple encoding avoids ambiguous string concatenation of signer/session IDs.
    group_ids = {pair: index for index, pair in enumerate(sorted(set(groups)))}
    return np.asarray([group_ids[pair] for pair in groups])


def split_indices(labels, groups=None, seed=42):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    indices = np.arange(len(labels))
    if groups is None:
        train_val, test = train_test_split(indices, test_size=0.15,
                                          stratify=labels, random_state=seed)
        train, val = train_test_split(train_val, test_size=0.15 / 0.85,
                                     stratify=labels[train_val], random_state=seed)
    else:
        if len(groups) != len(labels) or len(np.unique(groups)) < 3:
            raise ValueError("Session splitting needs at least three complete recording sessions")
        # Try reproducible group splits until each partition contains every class.
        for attempt in range(100):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + attempt)
            train_val, test = next(splitter.split(indices, labels, groups))
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed + attempt)
            a, b = next(splitter.split(train_val, labels[train_val], groups[train_val]))
            train, val = train_val[a], train_val[b]
            if all(set(labels[part]) == set(range(26)) for part in (train, val, test)):
                break
        else:
            raise ValueError("Cannot create disjoint sessions with all 26 classes; collect more sessions")
        assert not set(groups[train]) & set(groups[val])
        assert not set(groups[train]) & set(groups[test])
        assert not set(groups[val]) & set(groups[test])
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    assert len(train) + len(val) + len(test) == len(labels)
    return train, val, test
