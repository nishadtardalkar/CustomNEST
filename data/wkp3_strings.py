# data/wkp3_loader.py
import numpy as np



def scan_vocab(train_path, test_path=None):
    ents, rels = set(), set()
    for path in filter(None, [train_path, test_path]):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if len(parts) < 4: 
                    continue
                r, e1, e2, e3 = parts[:4]
                r  = _parse_token(r);  e1 = _parse_token(e1)
                e2 = _parse_token(e2); e3 = _parse_token(e3)
                # collect vocab
                rels.add(r); ents.update([e1, e2, e3])
    # map strings to ints if needed
    rels_sorted = sorted(rels, key=str)
    ents_sorted = sorted(ents, key=str)
    rel2id = {r:i for i,r in enumerate(rels_sorted)}
    ent2id = {e:i for i,e in enumerate(ents_sorted)}
    return ent2id, rel2id

def load_split(path, ent2id, rel2id):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 4: 
                continue
            r, e1, e2, e3 = parts[:4]
            r  = _parse_token(r);  e1 = _parse_token(e1)
            e2 = _parse_token(e2); e3 = _parse_token(e3)
            if (r in rel2id) and (e1 in ent2id) and (e2 in ent2id) and (e3 in ent2id):
                rows.append([rel2id[r], ent2id[e1], ent2id[e2], ent2id[e3]])  # (r,e1,e2,e3)
    ind = np.asarray(rows, dtype=np.int32)
    y   = np.ones((ind.shape[0], 1), dtype=np.float32)
    return ind, y

def build_all_true_heads(ind, r_slot=0, e1_slot=1, e2_slot=2, e3_slot=3):
    """(r,e2,e3) -> set of true e1 ids, using the provided slots."""
    D = {}
    if ind is None or ind.size == 0:
        return D
    for row in ind:
        r  = int(row[r_slot]); e1 = int(row[e1_slot])
        e2 = int(row[e2_slot]); e3 = int(row[e3_slot])
        D.setdefault((r, e2, e3), set()).add(e1)
    return D
def _parse_token(tok):
    # Keep Q/P ids as strings (relations/entities)
    if tok and tok[0] in "PQ":
        return tok
    # Treat timestamps (e.g., +1978-00-00T...) as categorical tokens.
    # Option A: keep full timestamp string
    if tok and tok[0] == '+':
        return tok
    # Be robust: fallback to string if it isn't a clean int
    try:
        return int(tok)
    except Exception:
        return tok
