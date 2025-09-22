# data/wikipeople3_utils.py
import numpy as np

def build_global_mappings_wkp3(train_path, test_path):
    ents, rels = set(), set()
    for path in [train_path, test_path]:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if not parts:
                    continue
                # Expect 3 tokens (h r t); ignore any label/score column if present
                h, r, t = parts[:3]
                ents.add(h); ents.add(t); rels.add(r)
    ent2id = {e:i for i,e in enumerate(sorted(ents))}
    rel2id = {r:i for i,r in enumerate(sorted(rels))}
    return ent2id, rel2id

def load_wikipeople3(path, ent2id, rel2id):
    rows_raw, rows = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if not parts:
                continue
            h, r, t = parts[:3]
            rows_raw.append((h, r, t))
            rows.append([ent2id[h], rel2id[r], ent2id[t]])
    ind = np.array(rows, dtype=np.int32)
    y   = np.ones((ind.shape[0], 1), dtype=np.float32)  # binary presence
    ind_raw = np.array(rows_raw, dtype=object)
    return ind, y, ind_raw

def build_all_true_tails(train_ind, valid_ind=None, test_ind=None):
    """
    Returns dict: key=(h,r) -> set of all true tails across splits (filtered protocol).
    """
    all_arrays = [a for a in [train_ind, valid_ind, test_ind] if a is not None]
    D = {}
    for A in all_arrays:
        for h, r, t in A:
            D.setdefault((int(h), int(r)), set()).add(int(t))
    return D
