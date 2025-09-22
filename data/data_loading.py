import numpy as np


def _load_data(data_path: str):
    ind = []
    y = []
    with open(data_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y.append(float(items[-1]))
            ind.append([int(idx)-1 for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)
    return ind, y

import numpy as np

def __load_data(path, sep=None, use_time=False, time_bins=0):
    inds, ys = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # auto-detect separator unless given
            s = sep
            if s is None:
                s = "\t" if ("\t" in line) else ("," if "," in line else " ")

            parts = line.split(s)

            if s == "\t":  # MovieLens 100K: user  item  rating  timestamp
                u = int(parts[0]) - 1
                i = int(parts[1]) - 1
                r = float(parts[2])
                if use_time and len(parts) >= 4 and time_bins > 0:
                    ts = int(parts[3])
                    # simple binning by quantiles or fixed width; here fixed width example:
                    # adjust as needed
                    t = ts  # or bucket(ts)
                    inds.append([u, i, t])
                else:
                    inds.append([u, i])
                ys.append(r)
            else:
                # original CSV behavior: last item is label
                *idx, y = parts
                inds.append([int(x) for x in idx])
                ys.append(float(y))

    ind = np.asarray(inds, dtype=np.int64)
    y = np.asarray(ys, dtype=np.float32)
    return ind, y
# data/data_loading.py
import re
import numpy as np

def _smart_split(line: str):
    # Split on one or more commas or whitespace
    return re.split(r'[,\s]+', line.strip())

def load_data_(path):
    """
    Reads triples from `path`.
    Supports:
      - comma-separated:  "30,1,1,1.9459"  or "30,1,1"
      - space/tabs:       "30 1 1 1.9459"  or "30 1 1"
    Returns:
      ind      : np.int32 shape [N,3]    (0-based indices for model)
      y        : np.float32 shape [N] or None
      ind_raw  : np.int32 shape [N,3]    (original IDs as in file)
    """
    ind_raw_list = []
    y_list = []

    with open(path, 'r') as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = _smart_split(line)

            # Basic validation: need at least h r t
            if len(parts) < 3:
                # skip lines that don't look like triples
                continue

            # detect header/non-numeric lines
            try:
                # try parse the first three as ints
                h_raw = int(parts[0])
                r_raw = int(parts[1])
                t_raw = int(parts[2])
            except ValueError:
                # looks like a header row -> skip
                continue

            ind_raw_list.append([h_raw, r_raw, t_raw])

            # if we have >3 fields, try to parse the last one as label
            if len(parts) >= 4:
                try:
                    y_val = float(parts[-1])
                    y_list.append(y_val)
                except ValueError:
                    # non-numeric tail field -> treat as no label
                    pass

    if not ind_raw_list:
        raise ValueError(f"No triples found in {path}")

    ind_raw = np.array(ind_raw_list, dtype=np.int32)

    # Build per-mode vocabularies to reindex to 0..(n_k-1)
    # This makes no assumptions about contiguity or starting at 1.
    N = ind_raw.shape[0]
    nmod = 3
    ind = np.zeros_like(ind_raw, dtype=np.int32)
    nvec = []

    for k in range(nmod):
        # unique original ids for mode k
        uniques = np.unique(ind_raw[:, k])
        # map orig -> new 0-based
        orig2new = {orig: i for i, orig in enumerate(uniques)}
        # fill reindexed col
        ind[:, k] = np.vectorize(orig2new.get, otypes=[np.int32])(ind_raw[:, k])
        nvec.append(len(uniques))

    nvec = np.array(nvec, dtype=np.int32)

    # Labels
    y = None
    if len(y_list) == N:
        y = np.array(y_list, dtype=np.float32)
    elif len(y_list) == 0:
        y = None  # link prediction w/o labels
    else:
        # mixed lines: some with label, some without -> align or raise
        raise ValueError(
            f"Inconsistent labels in {path}: found {len(y_list)} labels for {N} triples."
        )

    return ind, y, ind_raw

# data/fb15k_utils.py
import numpy as np

def read_triples_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            h, r, t = line.split("\t")
            rows.append((h, r, t))
    return rows

def build_global_mappings(train_path, valid_path=None, test_path=None):
    all_rows = []
    all_rows += read_triples_tsv(train_path)
    if valid_path: all_rows += read_triples_tsv(valid_path)
    if test_path:  all_rows += read_triples_tsv(test_path)

    entities = sorted({h for h,_,_ in all_rows} | {t for _,_,t in all_rows})
    relations = sorted({r for _,r,_ in all_rows})

    ent2id = {e:i for i,e in enumerate(entities)}
    rel2id = {r:i for i,r in enumerate(relations)}
    return ent2id, rel2id


# data/data_loading.py
import numpy as np
from .fb15k_utils import read_triples_tsv

# data/data_loading.py
import numpy as np
from .fb15k_utils import read_triples_tsv

def load_data(path, ent2id=None, rel2id=None):
    rows = read_triples_tsv(path)

    # If maps not provided, build from this file alone
    if ent2id is None or rel2id is None:
        ents = sorted({h for h,_,_ in rows} | {t for _,_,t in rows})
        rels = sorted({r for _,r,_ in rows})
        ent2id = {e:i for i,e in enumerate(ents)}
        rel2id = {r:i for i,r in enumerate(rels)}

    print(f"[LOG] Loaded {len(rows)} triples from {path}")
    print(f"[LOG] #Entities = {len(ent2id)}, #Relations = {len(rel2id)}")

    H = np.fromiter((ent2id[h] for (h,_,_) in rows), dtype=np.int64)
    R = np.fromiter((rel2id[r] for (_,r,_) in rows), dtype=np.int64)
    T = np.fromiter((ent2id[t] for (_,_,t) in rows), dtype=np.int64)

    ind = np.stack([H, R, T], axis=1).astype(np.int32)
    y = np.ones((ind.shape[0], 1), dtype=np.float32)
    ind_raw = np.array(rows, dtype=object)  # (N,3) strings

    return ind, y, ind_raw, ent2id, rel2id

