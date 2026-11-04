import sys
import time
import numpy as np
import scipy.linalg as sla

rng = np.random.default_rng(2026)


def lu_sin_pivoteo(A):
    A = A.copy()
    n = A.shape[0]
    for k in range(n - 1):
        A[k + 1:, k] /= A[k, k]
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])
    return np.tril(A, -1) + np.eye(n), np.triu(A)


def lu_pivoteo_parcial(A):
    A = A.copy()
    n = A.shape[0]
    p = np.arange(n)
    for k in range(n - 1):
        m = k + np.argmax(np.abs(A[k:, k]))
        if m != k:
            A[[k, m], :] = A[[m, k], :]
            p[[k, m]] = p[[m, k]]
        A[k + 1:, k] /= A[k, k]
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])
    return np.tril(A, -1) + np.eye(n), np.triu(A), p


def generar(n, kappa, tipo):
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    Q = Q * np.sign(np.diag(R))
    lam = np.logspace(0, np.log10(kappa), n)
    if tipo == "indef":
        lam = lam * rng.choice([-1.0, 1.0], n)
    A = (Q * lam) @ Q.T
    return (A + A.T) / 2


def medir(f):
    t = time.perf_counter()
    x = f()
    return x, time.perf_counter() - t


def metodos(A):
    def lu_np():
        L, U = lu_sin_pivoteo(A)
        return lambda b: sla.solve_triangular(U, sla.solve_triangular(L, b, lower=True, unit_diagonal=True))

    def lu_pp():
        L, U, p = lu_pivoteo_parcial(A)
        return lambda b: sla.solve_triangular(U, sla.solve_triangular(L, b[p], lower=True, unit_diagonal=True))

    def inv():
        Ai = np.linalg.inv(A)
        return lambda b: Ai @ b

    def lapack():
        lu, piv = sla.lu_factor(A)
        return lambda b: sla.lu_solve((lu, piv), b)

    def qr():
        Q, R = sla.qr(A)
        return lambda b: sla.solve_triangular(R, Q.T @ b)

    return {"LU sin pivoteo": lu_np, "LU pivoteo parcial": lu_pp, "inv(A)": inv, "LAPACK lu": lapack, "QR": qr}


def main():
    tamanos = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [512, 1024, 2048]
    kappas = [1e2, 1e6, 1e10]
    tipos = ["spd", "indef"]
    eps = 1e-8
    print("tipo,n,kappa,metodo,t_fact,t_solve,err_rel,res_rel,err_rel_pert,res_rel_pert,amplif")
    for tipo in tipos:
        for n in tamanos:
            for kappa in kappas:
                A = generar(n, kappa, tipo)
                xe = np.ones(n)
                b = A @ xe
                dp = rng.standard_normal(n)
                bp = b + eps * np.linalg.norm(b) * dp / np.linalg.norm(dp)
                nA = np.linalg.norm(A, 2)
                for nombre, fact in metodos(A).items():
                    try:
                        solver, tf = medir(fact)
                        x, ts = medir(lambda: solver(b))
                        xp = solver(bp)
                    except Exception as e:
                        print(f"{tipo},{n},{kappa:.0e},{nombre},fallo:{type(e).__name__}")
                        continue
                    err = np.linalg.norm(x - xe) / np.linalg.norm(xe)
                    res = np.linalg.norm(b - A @ x) / (nA * np.linalg.norm(x) + np.linalg.norm(b))
                    errp = np.linalg.norm(xp - xe) / np.linalg.norm(xe)
                    resp = np.linalg.norm(bp - A @ xp) / (nA * np.linalg.norm(xp) + np.linalg.norm(bp))
                    amp = np.linalg.norm(xp - x) / np.linalg.norm(x) / eps
                    print(f"{tipo},{n},{kappa:.0e},{nombre},{tf:.4f},{ts:.5f},{err:.3e},{res:.3e},{errp:.3e},{resp:.3e},{amp:.3e}", flush=True)


main()
