# data/fb15k_utils.py
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
    if valid_path:
        all_rows += read_triples_tsv(valid_path)
    if test_path:
        all_rows += read_triples_tsv(test_path)

    entities = sorted({h for h, _, _ in all_rows} | {t for _, _, t in all_rows})
    relations = sorted({r for _, r, _ in all_rows})

    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    return ent2id, rel2id
