import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import MATRICES
from tabla_ej2 import directo, cg

d = directo()
c = cg()
ordenes = [("natural", "Natural"), ("rcm", "RCM"), ("amd", "MMD (SuperLU)"), ("amd_real", "AMD (SuiteSparse)")]
x = np.arange(len(MATRICES))
w = 0.2


def barras(col, ylabel, archivo, tope=None):
    fig, a = plt.subplots(figsize=(9.5, 3.8))
    for i, (o, etiqueta) in enumerate(ordenes):
        v = []
        for m in MATRICES:
            f = d[(d.matriz == m) & (d.orden == o)].iloc[0]
            if f.estado == "ok":
                v.append(f[col])
            else:
                v.append(np.nan)
        a.bar(x + (i - 1.5) * w, v, w, label=etiqueta)
    a.set_yscale("log")
    a.set_xticks(x)
    a.set_xticklabels(MATRICES, rotation=30, ha="right")
    a.set_ylabel(ylabel)
    a.grid(True, axis="y", which="both", alpha=0.3)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(archivo, dpi=200)


barras("t_fact", "Tiempo de factorización (s)", "figs/directo_tiempo.png")
barras("rss_MB", "Memoria pico (MB)", "figs/directo_memoria.png")

fig, a = plt.subplots(figsize=(9.5, 3.8))
for i, (p, etiqueta) in enumerate([("sin precond.", "Sin precondicionador"), ("Jacobi", "Jacobi"), ("ILU0", "ILU(0)")]):
    v = []
    for m in MATRICES:
        f = c[(c.matriz == m) & (c.precond == p)].iloc[0]
        v.append(f.iter if f.iter > 0 else np.nan)
    a.bar(x + (i - 1) * w, v, w, label=etiqueta)
a.set_yscale("log")
a.set_xticks(x)
a.set_xticklabels(MATRICES, rotation=30, ha="right")
a.set_ylabel("Iteraciones de CG")
a.grid(True, axis="y", which="both", alpha=0.3)
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/cg_iteraciones.png", dpi=200)
