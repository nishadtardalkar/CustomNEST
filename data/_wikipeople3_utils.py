# data/wikipeople3_utils.py
import numpy as np

def _read_rows_4col(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # Keep the first 4 whitespace-separated tokens as-is
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows

def read_wp3(path):
    """List of (rel, e1, e2, e3) strings."""
    return _read_rows_4col(path)

def build_wp3_mappings(train_path, valid_path=None, test_path=None):
    """Builds vocabularies for:
       - mode0: entity2
       - mode1: composite (relation|entity3)
       - mode2: entity1 (predicted)
    """
    all_rows = []
    for p in (train_path, valid_path, test_path):
        if p:
            all_rows.extend(read_wp3(p))

    e2_set, re3_set, e1_set = set(), set(), set()
    for rel, e1, e2, e3 in all_rows:
        e2_set.add(e2)
        re3_set.add(f"{rel}|{e3}")
        e1_set.add(e1)

    e2_to_id  = {tok: i for i, tok in enumerate(sorted(e2_set))}
    re3_to_id = {tok: i for i, tok in enumerate(sorted(re3_set))}
    e1_to_id  = {tok: i for i, tok in enumerate(sorted(e1_set))}
    return e2_to_id, re3_to_id, e1_to_id

def encode_wp3_as_triples(rows, e2_to_id, re3_to_id, e1_to_id):
    """Return int32 (N,3) with:
       col0 = id(entity2), col1 = id(f"{relation}|{entity3}"), col2 = id(entity1)
    """
    H, R, T = [], [], []
    for rel, e1, e2, e3 in rows:
        H.append(e2_to_id[e2])
        R.append(re3_to_id[f"{rel}|{e3}"])
        T.append(e1_to_id[e1])
    return np.stack([H, R, T], axis=1).astype(np.int32)
