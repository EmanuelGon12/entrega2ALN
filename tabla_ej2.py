import pandas as pd


def directo(archivo="ej2_directo.csv"):
    d = pd.read_csv(archivo)
    for c in d.columns:
        if c not in ("matriz", "orden", "estado"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def cg(archivo="ej2_cg.csv"):
    return pd.read_csv(archivo)
