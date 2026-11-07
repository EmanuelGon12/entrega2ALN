import subprocess
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

from comun import MATRICES, cargar, sistema, residuo_rel

RTOL = 1e-12
MAXITER = 20000
REPS = 3
LIMITE_REPS = 1e9
TIMEOUT_LAM = 300


@njit(cache=True)
def ilu0_numba(n, indptr, indices, data):
    w = data.copy()
    dptr = np.full(n, -1, np.int64)
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            if indices[p] == i:
                dptr[i] = p
    pos = np.full(n, -1, np.int64)
    for i in range(n):
        if dptr[i] < 0:
            return w, dptr, i
        for p in range(indptr[i], indptr[i + 1]):
            pos[indices[p]] = p
        for p in range(indptr[i], indptr[i + 1]):
            k = indices[p]
            if k >= i:
                break
            w[p] /= w[dptr[k]]
            lik = w[p]
            for q in range(dptr[k] + 1, indptr[k + 1]):
                pp = pos[indices[q]]
                if pp >= 0:
                    w[pp] -= lik * w[q]
        for p in range(indptr[i], indptr[i + 1]):
            pos[indices[p]] = -1
        if w[dptr[i]] <= 0.0:
            return w, dptr, i
    return w, dptr, -1


@njit(cache=True)
def ilu0_apply(n, indptr, indices, w, dptr, r):
    y = r.copy()
    for i in range(n):
        s = y[i]
        for p in range(indptr[i], dptr[i]):
            s -= w[p] * y[indices[p]]
        y[i] = s
    for i in range(n - 1, -1, -1):
        s = y[i]
        for p in range(dptr[i] + 1, indptr[i + 1]):
            s -= w[p] * y[indices[p]]
        y[i] = s / w[dptr[i]]
    return y


def cg(A, b, M=None):
    it = [0]

    def cb(_):
        it[0] += 1

    t0 = time.perf_counter()
    x, info = spla.cg(A, b, rtol=RTOL, atol=0.0, maxiter=MAXITER, M=M, callback=cb)
    return x, info, it[0], time.perf_counter() - t0


def calentar():
    m = 50
    T = sp.diags([-np.ones(m - 1), 2.5 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1], format="csr")
    T.sort_indices()
    ip, ind = T.indptr.astype(np.int64), T.indices.astype(np.int64)
    w, dptr, _ = ilu0_numba(m, ip, ind, T.data)
    ilu0_apply(m, ip, ind, w, dptr, np.ones(m))


def una_pasada(nombre, A, b, xe):
    n = A.shape[0]
    filas = []

    def error(x):
        return float(np.linalg.norm(x - xe) / np.linalg.norm(xe))

    x, info, it, t = cg(A, b)
    filas.append([nombre, "sin precond.", 0.0, it, t, residuo_rel(A, x, b), info, error(x)])

    t0 = time.perf_counter()
    d = 1.0 / A.diagonal()
    Mj = spla.LinearOperator((n, n), matvec=lambda r: d * r, dtype=np.float64)
    ts = time.perf_counter() - t0
    x, info, it, t = cg(A, b, Mj)
    filas.append([nombre, "Jacobi", ts, it, t, residuo_rel(A, x, b), info, error(x)])

    ip, ind = A.indptr.astype(np.int64), A.indices.astype(np.int64)
    t0 = time.perf_counter()
    w, dptr, fallo = ilu0_numba(n, ip, ind, A.data)
    ts = time.perf_counter() - t0
    if fallo >= 0:
        filas.append([nombre, "ILU0", ts, -1, 0.0, float("nan"), -2, float("nan")])
    else:
        Mi = spla.LinearOperator((n, n), matvec=lambda r: ilu0_apply(n, ip, ind, w, dptr, r), dtype=np.float64)
        x, info, it, t = cg(A, b, Mi)
        filas.append([nombre, "ILU0", ts, it, t, residuo_rel(A, x, b), info, error(x)])
    return filas


def correr(nombre):
    A = cargar(nombre).tocsr()
    A.sort_indices()
    b, xe = sistema(A)
    pasadas = [una_pasada(nombre, A, b, xe)]
    total = sum(f[2] + f[4] for f in pasadas[0])
    if total < LIMITE_REPS:
        pasadas += [una_pasada(nombre, A, b, xe) for _ in range(REPS - 1)]
    salida = []
    for k, f in enumerate(pasadas[0]):
        fila = list(f)
        fila[2] = float(np.median([p[k][2] for p in pasadas]))
        fila[4] = float(np.median([p[k][4] for p in pasadas]))
        salida.append(fila + [len(pasadas)])
    return salida


def lam(nombre, que):
    A = cargar(nombre).tocsr()
    if que == "max":
        v = spla.eigsh(A, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0]
    else:
        lu = spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        inv = spla.LinearOperator(A.shape, matvec=lu.solve, dtype=np.float64)
        v = spla.eigsh(A, k=1, sigma=0, which="LM", OPinv=inv, return_eigenvectors=False)[0]
    print("LAM" + repr(float(v)), flush=True)


def autovalor(nombre, que):
    try:
        out = subprocess.run([sys.executable, __file__, "--lam", que, nombre], capture_output=True, text=True, timeout=TIMEOUT_LAM)
    except subprocess.TimeoutExpired:
        print(f"aviso: autovalor {que} de {nombre} excedió {TIMEOUT_LAM} s", file=sys.stderr, flush=True)
        return float("nan")
    for l in out.stdout.splitlines():
        if l.startswith("LAM"):
            return float(l[3:])
    print(f"aviso: autovalor {que} de {nombre} falló\n{out.stderr[-500:]}", file=sys.stderr, flush=True)
    return float("nan")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--lam":
        lam(sys.argv[3], sys.argv[2])
        return
    nombres = sys.argv[1].split(",") if len(sys.argv) > 1 else MATRICES
    calentar()
    print("matriz,precond,t_setup,iter,t_cg,res_rel,info,err_rel,reps,kappa_est,lmin,lmax", flush=True)
    for nombre in nombres:
        lmax = autovalor(nombre, "max")
        lmin = autovalor(nombre, "min")
        kappa = lmax / lmin if lmin > 0 else float("nan")
        for f in correr(nombre):
            print(f"{f[0]},{f[1]},{f[2]:.4g},{f[3]},{f[4]:.4g},{f[5]:.3e},{f[6]},{f[7]:.3e},{f[8]},{kappa:.4e},{lmin:.4e},{lmax:.4e}", flush=True)


if __name__ == "__main__":
    main()
