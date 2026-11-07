import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

from comun import MATRICES, cargar, sistema, residuo_rel

RTOL = 1e-12
MAXITER = 20000


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


def correr(nombre):
    A = cargar(nombre).tocsr()
    A.sort_indices()
    b, _ = sistema(A)
    n = A.shape[0]
    filas = []

    x, info, it, t = cg(A, b)
    filas.append((nombre, "sin precond.", 0.0, it, t, residuo_rel(A, x, b), info))

    t0 = time.perf_counter()
    d = 1.0 / A.diagonal()
    Mj = spla.LinearOperator((n, n), matvec=lambda r: d * r, dtype=np.float64)
    ts = time.perf_counter() - t0
    x, info, it, t = cg(A, b, Mj)
    filas.append((nombre, "Jacobi", ts, it, t, residuo_rel(A, x, b), info))

    t0 = time.perf_counter()
    w, dptr, fallo = ilu0_numba(n, A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data)
    ts = time.perf_counter() - t0
    if fallo >= 0:
        filas.append((nombre, "ILU0", ts, -1, 0.0, float("nan"), -2))
    else:
        ip, ind = A.indptr.astype(np.int64), A.indices.astype(np.int64)
        Mi = spla.LinearOperator((n, n), matvec=lambda r: ilu0_apply(n, ip, ind, w, dptr, r), dtype=np.float64)
        x, info, it, t = cg(A, b, Mi)
        filas.append((nombre, "ILU0", ts, it, t, residuo_rel(A, x, b), info))
    return filas


def main():
    nombres = sys.argv[1].split(",") if len(sys.argv) > 1 else MATRICES
    print("matriz,precond,t_setup,iter,t_cg,res_rel,info", flush=True)
    for nombre in nombres:
        for f in correr(nombre):
            print(f"{f[0]},{f[1]},{f[2]:.4g},{f[3]},{f[4]:.4g},{f[5]:.3e},{f[6]}", flush=True)


if __name__ == "__main__":
    main()
