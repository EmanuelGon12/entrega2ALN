import json
import resource
import subprocess
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import reverse_cuthill_mckee

from comun import MATRICES, cargar, sistema, residuo_rel

ORDENES = ["natural", "rcm", "amd"]
TIMEOUT = 300


def worker(nombre, orden):
    A = cargar(nombre)
    b, xe = sistema(A)
    n = A.shape[0]
    t0 = time.perf_counter()
    if orden == "rcm":
        p = reverse_cuthill_mckee(A, symmetric_mode=True)
        Ap = A[p][:, p].tocsc()
        spec = "NATURAL"
    elif orden == "amd":
        p = np.arange(n)
        Ap = A.tocsc()
        spec = "MMD_AT_PLUS_A"
    else:
        p = np.arange(n)
        Ap = A.tocsc()
        spec = "NATURAL"
    t_ord = time.perf_counter() - t0
    t0 = time.perf_counter()
    lu = spla.splu(Ap, permc_spec=spec, diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    t_fact = time.perf_counter() - t0
    t0 = time.perf_counter()
    y = lu.solve(b[p])
    x = np.empty(n)
    x[p] = y
    t_solve = time.perf_counter() - t0
    res = {
        "n": n,
        "nnz": int(A.nnz),
        "orden": orden,
        "t_orden": t_ord,
        "t_fact": t_fact,
        "t_solve": t_solve,
        "nnz_LU": int(lu.L.nnz + lu.U.nnz),
        "res_rel": residuo_rel(A, x, b),
        "err_rel": float(np.linalg.norm(x - xe) / np.linalg.norm(xe)),
        "rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    print("JSON" + json.dumps(res))


def main():
    if len(sys.argv) == 3:
        worker(sys.argv[1], sys.argv[2])
        return
    nombres = sys.argv[1].split(",") if len(sys.argv) == 2 else MATRICES
    cols = ["matriz", "orden", "n", "nnz", "t_orden", "t_fact", "t_solve", "nnz_LU", "res_rel", "err_rel", "rss_MB"]
    print(",".join(cols), flush=True)
    for nombre in nombres:
        for orden in ORDENES:
            try:
                out = subprocess.run([sys.executable, __file__, nombre, orden], capture_output=True, text=True, timeout=TIMEOUT)
                linea = [l for l in out.stdout.splitlines() if l.startswith("JSON")]
                if not linea:
                    print(f"{nombre},{orden},fallo", flush=True)
                    continue
                r = json.loads(linea[0][4:])
                print(",".join([nombre, orden] + [f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols[2:]]), flush=True)
            except subprocess.TimeoutExpired:
                print(f"{nombre},{orden},timeout", flush=True)


if __name__ == "__main__":
    main()
