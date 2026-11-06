import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

MATRICES = ["gyro_m", "msc10848", "bcsstk17", "cfd1", "cant", "bmwcra_1", "apache2", "ecology2", "thermal2"]
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrices")


def cargar(nombre):
    cache = os.path.join(DIR, nombre, nombre + ".npz")
    if os.path.exists(cache):
        return sp.load_npz(cache)
    A = sio.mmread(os.path.join(DIR, nombre, nombre + ".mtx")).tocsr()
    A = A.astype(np.float64)
    sp.save_npz(cache, A)
    return A


def sistema(A):
    x = np.ones(A.shape[0])
    return A @ x, x


def residuo_rel(A, x, b):
    return np.linalg.norm(b - A @ x) / np.linalg.norm(b)
