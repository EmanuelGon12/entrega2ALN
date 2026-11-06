import pandas as pd

COLS = ["matriz", "orden", "n", "nnz", "t_orden", "t_fact", "t_solve", "nnz_LU", "res_rel", "err_rel", "rss_MB"]


def directo():
    partes = [pd.read_csv(f, names=COLS, header=0, dtype=str) for f in ("ej2_directo.csv", "ej2_directo_b.csv")]
    d = pd.concat(partes, ignore_index=True)
    d["estado"] = d["n"].where(~d["n"].fillna("").str.isdigit(), "ok")
    base = d[d.estado == "ok"].groupby("matriz")[["n", "nnz"]].first()
    for c in ["t_orden", "t_fact", "t_solve", "nnz_LU", "res_rel", "err_rel", "rss_MB"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["n"] = d["matriz"].map(base["n"]).astype(float)
    d["nnz"] = d["matriz"].map(base["nnz"]).astype(float)
    return d


def cg():
    return pd.read_csv("ej2_cg.csv")
