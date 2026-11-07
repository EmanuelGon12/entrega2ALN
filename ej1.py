import sys
import time
import numpy as np
import scipy.linalg as sla
from scipy.linalg import lapack
from threadpoolctl import threadpool_info

rng = np.random.default_rng(2026)
REPS = 3


def lu_sin_pivoteo(A):
    A = A.copy()
    n = A.shape[0]
    for k in range(n - 1):
        if A[k, k] == 0.0:
            raise ZeroDivisionError(f"pivote nulo en k={k}")
        A[k + 1:, k] /= A[k, k]
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])
    if A[n - 1, n - 1] == 0.0:
        raise ZeroDivisionError(f"pivote nulo en k={n - 1}")
    return np.tril(A, -1) + np.eye(n), np.triu(A)


def lu_pivoteo_parcial(A):
    A = A.copy()
    n = A.shape[0]
    p = np.arange(n)
    swaps = 0
    for k in range(n - 1):
        m = k + np.argmax(np.abs(A[k:, k]))
        if m != k:
            A[[k, m], :] = A[[m, k], :]
            p[[k, m]] = p[[m, k]]
            swaps += 1
        if A[k, k] == 0.0:
            raise ZeroDivisionError(f"pivote nulo en k={k}")
        A[k + 1:, k] /= A[k, k]
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])
    if A[n - 1, n - 1] == 0.0:
        raise ZeroDivisionError(f"pivote nulo en k={n - 1}")
    return np.tril(A, -1) + np.eye(n), np.triu(A), p, swaps


def generar(n, kappa, tipo):
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    Q = Q * np.sign(np.diag(R))
    lam = np.logspace(0, np.log10(kappa), n)
    if tipo == "indef":
        lam = lam * rng.choice([-1.0, 1.0], n)
    A = (Q * lam) @ Q.T
    return (A + A.T) / 2


def mediana(f):
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t)
    return r, float(np.median(ts))


def metodos(A, extras):
    nA = np.abs(A).max()

    def lu_np():
        L, U = lu_sin_pivoteo(A)
        extras["LU sin pivoteo"] = ("", np.abs(U).max() / nA)
        return lambda b: sla.solve_triangular(U, sla.solve_triangular(L, b, lower=True, unit_diagonal=True))

    def lu_pp():
        L, U, p, swaps = lu_pivoteo_parcial(A)
        extras["LU pivoteo parcial"] = (swaps, np.abs(U).max() / nA)
        return lambda b: sla.solve_triangular(U, sla.solve_triangular(L, b[p], lower=True, unit_diagonal=True))

    def inv():
        Ai = np.linalg.inv(A)
        return lambda b: Ai @ b

    def lapack_lu():
        lu, piv = sla.lu_factor(A)
        return lambda b: sla.lu_solve((lu, piv), b)

    def qr():
        Q, R = sla.qr(A)
        return lambda b: sla.solve_triangular(R, Q.T @ b)

    def qr_raw():
        (qr_, tau), R = sla.qr(A, mode="raw")
        R = np.triu(R)

        def resolver(b):
            c, work, info = lapack.dormqr("L", "T", qr_, tau, b.reshape(-1, 1).copy(order="F"), lwork=-1)
            c, work, info = lapack.dormqr("L", "T", qr_, tau, b.reshape(-1, 1).copy(order="F"), lwork=int(work[0]))
            return sla.solve_triangular(R, c[:, 0])

        return resolver

    return {
        "LU sin pivoteo": lu_np,
        "LU pivoteo parcial": lu_pp,
        "inv(A)": inv,
        "LAPACK lu": lapack_lu,
        "QR": qr,
        "QR (sin formar Q)": qr_raw,
    }


def main():
    print(threadpool_info(), file=sys.stderr)
    tamanos = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [512, 1024, 2048]
    kappas = [1e2, 1e6, 1e10]
    tipos = ["spd", "indef"]
    eps = 1e-8
    print("tipo,n,kappa,metodo,t_fact,t_solve,err_rel,res_rel,err_rel_pert,res_rel_pert,amplif,n_swaps,rho")
    for tipo in tipos:
        for n in tamanos:
            for kappa in kappas:
                A = generar(n, kappa, tipo)
                xe = np.ones(n)
                b = A @ xe
                dp = rng.standard_normal(n)
                bp = b + eps * np.linalg.norm(b) * dp / np.linalg.norm(dp)
                nA = np.linalg.norm(A, 2)
                extras = {}
                sol = {}
                for nombre, fact in metodos(A, extras).items():
                    try:
                        solver, tf = mediana(fact)
                        x, ts = mediana(lambda: solver(b))
                        xp = solver(bp)
                    except ZeroDivisionError as e:
                        print(f"{tipo},{n},{kappa:.0e},{nombre},fallo:{e}", flush=True)
                        continue
                    sol[nombre] = x
                    err = np.linalg.norm(x - xe) / np.linalg.norm(xe)
                    res = np.linalg.norm(b - A @ x) / (nA * np.linalg.norm(x) + np.linalg.norm(b))
                    errp = np.linalg.norm(xp - xe) / np.linalg.norm(xe)
                    resp = np.linalg.norm(bp - A @ xp) / (nA * np.linalg.norm(xp) + np.linalg.norm(bp))
                    amp = np.linalg.norm(xp - x) / np.linalg.norm(x) / eps
                    sw, rho = extras.get(nombre, ("", ""))
                    rho = f"{rho:.4e}" if rho != "" else ""
                    print(f"{tipo},{n},{kappa:.0e},{nombre},{tf:.4f},{ts:.5f},{err:.3e},{res:.3e},{errp:.3e},{resp:.3e},{amp:.3e},{sw},{rho}", flush=True)
                if kappa == 1e2 and "QR" in sol and "QR (sin formar Q)" in sol:
                    d = np.linalg.norm(sol["QR"] - sol["QR (sin formar Q)"]) / np.linalg.norm(sol["QR"])
                    print(f"verificacion QR vs QR sin formar Q ({tipo}, n={n}): {d:.2e}", file=sys.stderr)


if __name__ == "__main__":
    main()
