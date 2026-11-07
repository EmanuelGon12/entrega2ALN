import json
import resource
import subprocess
import sys
import time
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import reverse_cuthill_mckee

from comun import MATRICES, cargar, sistema, residuo_rel

TIMEOUT = 300
COLS = ["matriz", "orden", "n", "nnz", "t_prep", "t_orden", "t_perm", "t_fact", "t_solve", "nnz_LU",
        "res_rel", "err_rel", "rss_MB", "banda", "env", "estado"]


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


def cholmod_disponible():
    try:
        import sksparse.cholmod
        return True
    except ImportError:
        return False


def ordenes_disponibles():
    ordenes = ["natural", "rcm", "amd"]
    if cholmod_disponible():
        ordenes.append("amd_cholmod")
    else:
        print("aviso: scikit-sparse no está instalado; se omite el orden amd_cholmod", file=sys.stderr)
    return ordenes


def worker(nombre, orden):
    A = cargar(nombre)
    b, xe = sistema(A)
    n = A.shape[0]
    res = {"n": n, "nnz": int(A.nnz), "orden": orden, "t_orden": 0.0, "t_perm": 0.0}
    p = np.arange(n)
    spec = "NATURAL"
    if orden in ("rcm", "amd_cholmod"):
        t0 = time.perf_counter()
        if orden == "rcm":
            p = reverse_cuthill_mckee(A, symmetric_mode=True)
        else:
            from sksparse.cholmod import analyze
            p = np.asarray(analyze(A.tocsc(), ordering_method="amd").P())
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


def fila(nombre, orden, r, estado):
    r = dict(r, estado=estado)
    vals = []
    for c in COLS[2:]:
        v = r.get(c)
        vals.append("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)))
    return ",".join([nombre, orden] + vals)


def texto(x):
    return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")


def main():
    if len(sys.argv) == 3:
        worker(sys.argv[1], sys.argv[2])
        return
    nombres = sys.argv[1].split(",") if len(sys.argv) == 2 else MATRICES
    ordenes = ordenes_disponibles()
    print(",".join(COLS), flush=True)
    for nombre in nombres:
        for orden in ordenes:
            try:
                out = subprocess.run([sys.executable, __file__, nombre, orden], capture_output=True, text=True, timeout=TIMEOUT)
                r = ultimo(out.stdout, "JSON")
                if r is not None:
                    print(fila(nombre, orden, r, "ok"), flush=True)
                    continue
                estado = "fallo"
                parcial = ultimo(out.stdout, "PARCIAL") or {}
                print(f"matriz={nombre} orden={orden} codigo={out.returncode}\n{out.stderr[-2000:]}", file=sys.stderr, flush=True)
            except subprocess.TimeoutExpired as e:
                estado = "timeout"
                parcial = ultimo(texto(e.stdout), "PARCIAL") or {}
                print(f"matriz={nombre} orden={orden} timeout={TIMEOUT}s\n{texto(e.stderr)[-2000:]}", file=sys.stderr, flush=True)
            print(fila(nombre, orden, parcial, estado), flush=True)


if __name__ == "__main__":
    main()
