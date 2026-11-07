import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("ej1_resultados.csv")
d["kappa"] = d["kappa"].astype(float)
metodos = ["LU sin pivoteo", "LU pivoteo parcial", "inv(A)", "LAPACK lu", "QR", "QR (sin formar Q)"]
marc = dict(zip(metodos, "osv^DP"))
tipos = {"spd": "Matrices SPD", "indef": "Matrices simétricas indefinidas"}

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, (t, tt) in zip(ax, tipos.items()):
    for m in metodos:
        s = d[(d.tipo == t) & (d.n == 2048) & (d.metodo == m)]
        a.loglog(s.kappa, s.err_rel, marker=marc[m], label=m)
    a.set_title(tt, fontsize=10)
    a.set_xlabel(r"$\kappa(A)$")
    a.grid(True, which="both", alpha=0.3)
ax[0].set_ylabel("Error relativo en $x$")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/error_vs_kappa.png", dpi=200)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, (t, tt) in zip(ax, tipos.items()):
    for m in metodos:
        s = d[(d.tipo == t) & (d.n == 2048) & (d.metodo == m)]
        a.loglog(s.kappa, s.res_rel, marker=marc[m], label=m)
    a.set_title(tt, fontsize=10)
    a.set_xlabel(r"$\kappa(A)$")
    a.grid(True, which="both", alpha=0.3)
ax[0].set_ylabel("Residuo relativo")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/residuo_vs_kappa.png", dpi=200)

fig, a = plt.subplots(figsize=(5.2, 3.6))
for m in metodos:
    s = d[(d.tipo == "spd") & (d.kappa == 1e2) & (d.metodo == m)]
    a.loglog(s.n, s.t_fact, marker=marc[m], label=m)
a.set_xlabel("n")
a.set_ylabel("Tiempo de factorización (s)")
a.set_xticks([512, 1024, 2048])
a.set_xticklabels(["512", "1024", "2048"])
a.grid(True, which="both", alpha=0.3)
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/tiempo_vs_n.png", dpi=200)
