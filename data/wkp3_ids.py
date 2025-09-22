# data/wkp3_ids.py
import numpy as np

def load_wkp3_ids(path, reorder_to_e1_first=True):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            r, e1, e2, e3 = map(int, parts[:4])
            if reorder_to_e1_first:
                rows.append([e1, r, e2, e3])  # (e1, r, e2, e3)
            else:
                rows.append([r, e1, e2, e3])  # (r, e1, e2, e3)

    ind = np.asarray(rows, dtype=np.int32)

    # If your data is 1-based (IDs start at 1), shift to 0-based:
    if ind.min() == 1:
        ind -= 1

    y = np.ones((ind.shape[0], 1), dtype=np.float32)
    return ind, y

def derive_nvec_4ary_e1_first(train_ind, test_ind=None):
    """
    For internal order (e1, r, e2, e3):
      nvec = [n_ent, n_rel, n_ent, n_ent]
    """
    Xs = [train_ind] + ([test_ind] if test_ind is not None else [])
    X  = np.vstack(Xs) if len(Xs) > 1 else train_ind
    n_ent = int(max(X[:,0].max(), X[:,2].max(), X[:,3].max()) + 1)
    n_rel = int(X[:,1].max() + 1)
    return np.array([n_ent, n_rel, n_ent, n_ent], dtype=np.int32)

def build_all_true_heads_e1_first(train_ind, valid_ind=None, test_ind=None):
    """
    Filter dict for head prediction when internal order is (e1, r, e2, e3).
    Key = (r, e2, e3) -> set of true e1.
    """
    D = {}
    for A in filter(lambda a: a is not None, [train_ind, valid_ind, test_ind]):
        for e1, r, e2, e3 in A:
            key = (int(r), int(e2), int(e3))
            D.setdefault(key, set()).add(int(e1))
    return D
