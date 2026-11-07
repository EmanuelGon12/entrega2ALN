import json
import os
import resource
import subprocess
import sys
import time
import psutil
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import reverse_cuthill_mckee

from comun import MATRICES, cargar, sistema, residuo_rel

TIMEOUT = 300
REPS = int(os.environ.get("REPS", "1"))
ORDEN_REAL = "amd_real"
COLS = ["matriz", "orden", "n", "nnz", "t_prep", "t_orden", "t_perm", "t_fact", "t_solve", "nnz_LU",
        "res_rel", "err_rel", "rss_MB", "banda", "env", "estado", "reps", "t_fact_min", "t_fact_max",
        "rss_min", "rss_max", "rss_obs_MB", "swap_MB"]


def banda_env(A):
    A = A.tocsr()
    A.sort_indices()
    n = A.shape[0]
    primero = A.indices[A.indptr[:-1]]
    d = np.maximum(np.arange(n) - primero, 0)
    return int(d.max()), int(d.sum())


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def amd_cvxopt(A):
    from cvxopt import spmatrix, amd
    C = A.tocoo()
    m = C.row >= C.col
    S = spmatrix(C.data[m], C.row[m].tolist(), C.col[m].tolist(), A.shape)
    return np.array(amd.order(S, uplo="L")).ravel()


def amd_sksparse(A):
    from sksparse.cholmod import analyze
    return np.asarray(analyze(A.tocsc(), ordering_method="amd").P())


def backend_amd():
    for nombre, f, modulo in (("sksparse", amd_sksparse, "sksparse.cholmod"), ("cvxopt", amd_cvxopt, "cvxopt.amd")):
        try:
            __import__(modulo)
            return nombre, f
        except ImportError:
            continue
    return None, None


def ordenes_disponibles():
    ordenes = ["natural", "rcm", "amd"]
    if backend_amd()[0] is not None:
        ordenes.append(ORDEN_REAL)
    else:
        print("aviso: no hay scikit-sparse ni cvxopt; se omite el orden amd_real", file=sys.stderr)
    return ordenes


def worker(nombre, orden):
    A = cargar(nombre)
    b, xe = sistema(A)
    n = A.shape[0]
    res = {"n": n, "nnz": int(A.nnz), "orden": orden, "t_orden": 0.0, "t_perm": 0.0}
    p = np.arange(n)
    spec = "NATURAL"
    if orden in ("rcm", ORDEN_REAL):
        t0 = time.perf_counter()
        if orden == "rcm":
            p = reverse_cuthill_mckee(A, symmetric_mode=True)
        else:
            p = backend_amd()[1](A)
        res["t_orden"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ap = A[p][:, p].tocsc()
        res["t_perm"] = time.perf_counter() - t0
        res["t_prep"] = 0.0
    else:
        t0 = time.perf_counter()
        Ap = A.tocsc()
        res["t_prep"] = time.perf_counter() - t0
        if orden == "amd":
            spec = "MMD_AT_PLUS_A"
    if orden != "amd":
        res["banda"], res["env"] = banda_env(Ap)
    print("PARCIAL" + json.dumps(res), flush=True)
    t0 = time.perf_counter()
    lu = spla.splu(Ap, permc_spec=spec, diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    res["t_fact"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    y = lu.solve(b[p])
    x = np.empty(n)
    x[p] = y
    res["t_solve"] = time.perf_counter() - t0
    res["nnz_LU"] = int(lu.L.nnz + lu.U.nnz)
    res["res_rel"] = residuo_rel(A, x, b)
    res["err_rel"] = float(np.linalg.norm(x - xe) / np.linalg.norm(xe))
    res["rss_MB"] = rss_mb()
    print("JSON" + json.dumps(res), flush=True)


def ultimo(salida, prefijo):
    lineas = [l for l in (salida or "").splitlines() if l.startswith(prefijo)]
    return json.loads(lineas[-1][len(prefijo):]) if lineas else None


def fmt(v):
    return "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def fila(nombre, orden, r, estado):
    r = dict(r, estado=estado)
    return ",".join([nombre, orden] + [fmt(r.get(c)) for c in COLS[2:]])


def ejecutar(nombre, orden):
    p = subprocess.Popen([sys.executable, __file__, nombre, orden], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.perf_counter()
    rss = swap = 0.0
    base = psutil.swap_memory().used / 2**20
    expiro = False
    while p.poll() is None:
        if time.perf_counter() - t0 > TIMEOUT:
            p.kill()
            expiro = True
            break
        try:
            rss = max(rss, psutil.Process(p.pid).memory_info().rss / 2**20)
        except psutil.Error:
            pass
        swap = max(swap, psutil.swap_memory().used / 2**20 - base)
        time.sleep(0.5)
    out, err = p.communicate()
    return out, err, p.returncode, expiro, rss, swap


def mediana(rs, c):
    return float(np.median([r[c] for r in rs]))


def main():
    if len(sys.argv) == 3:
        worker(sys.argv[1], sys.argv[2])
        return
    nombres = sys.argv[1].split(",") if len(sys.argv) == 2 else MATRICES
    ordenes = ordenes_disponibles()
    print(",".join(COLS), flush=True)
    for nombre in nombres:
        for orden in ordenes:
            rs = []
            rss_obs = swap_obs = 0.0
            estado = "ok"
            parcial = {}
            for _ in range(REPS):
                out, err, codigo, expiro, rss, swap = ejecutar(nombre, orden)
                rss_obs, swap_obs = max(rss_obs, rss), max(swap_obs, swap)
                r = ultimo(out, "JSON")
                if r is not None:
                    rs.append(r)
                    continue
                parcial = ultimo(out, "PARCIAL") or {}
                estado = "timeout" if expiro else "fallo"
                print(f"matriz={nombre} orden={orden} {estado} codigo={codigo} rss_obs={rss:.0f}MB swap={swap:.0f}MB\n{err[-2000:]}", file=sys.stderr, flush=True)
                break
            if not rs:
                parcial.update(rss_obs_MB=rss_obs, swap_MB=swap_obs, reps=0)
                print(fila(nombre, orden, parcial, estado), flush=True)
                continue
            r = dict(rs[0])
            for c in ("t_prep", "t_orden", "t_perm", "t_fact", "t_solve", "rss_MB"):
                r[c] = mediana(rs, c)
            r.update(reps=len(rs), t_fact_min=min(x["t_fact"] for x in rs), t_fact_max=max(x["t_fact"] for x in rs),
                     rss_min=min(x["rss_MB"] for x in rs), rss_max=max(x["rss_MB"] for x in rs), rss_obs_MB=rss_obs, swap_MB=swap_obs)
            print(fila(nombre, orden, r, estado), flush=True)


if __name__ == "__main__":
    main()
