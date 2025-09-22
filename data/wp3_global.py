# data/wp3_global.py
# Read (relation, e1, e2, e3) and build separate vocab for relations + a shared entity vocab.

import numpy as np
from collections import OrderedDict

def _read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split()
            assert len(parts) == 4, f"Expected 4 columns, got {len(parts)}: {line}"
            r, e1, e2, e3 = parts
            rows.append((r, e1, e2, e3))
    return rows

def _freeze_index(keys_iterable):
    od = OrderedDict()
    for k in keys_iterable:
        if k not in od:
            od[k] = len(od)
    return od

def build_global_vocab(train_path, valid_path, test_path):
    tr = _read_rows(train_path); va = _read_rows(valid_path); te = _read_rows(test_path)
    all_rows = tr + va + te
    rel2id = _freeze_index([r for (r,_,_,_) in all_rows])
    ent2id = _freeze_index([e for (_,e,_,_) in all_rows] +
                           [e for (_,_,e,_) in all_rows] +
                           [e for (_,_,_,e) in all_rows])
    return ent2id, rel2id

def encode_rows_as_quads(rows, ent2id, rel2id):
    # encode as [r_id, e1_id, e2_id, e3_id]
    out = np.zeros((len(rows), 4), dtype=np.int32)
    for i, (r, e1, e2, e3) in enumerate(rows):
        out[i, 0] = rel2id[r]
        out[i, 1] = ent2id[e1]
        out[i, 2] = ent2id[e2]
        out[i, 3] = ent2id[e3]
    return out
