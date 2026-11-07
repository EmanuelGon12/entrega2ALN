# Tarea 2 – Álgebra Lineal Numérica 2026

Ejercicio 1: sistemas densos (LU sin pivoteo, LU con pivoteo parcial, `inv(A)`, `lu` de LAPACK, QR).
Ejercicio 2: sistemas dispersos SPD de SuiteSparse (SuperLU con orden natural, RCM y MMD; gradiente conjugado
sin precondicionador, con Jacobi y con ILU(0) programado en Numba).

## Entorno

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Requiere SciPy ≥ 1.12 (el CG usa `rtol=`). Equipo de las mediciones: Apple M1 Pro, 10 núcleos, 16 GB de RAM,
macOS, Python 3.9.6. BLAS (`threadpoolctl.threadpool_info()`): OpenBLAS 0.3.27 incluido en la rueda de SciPy,
10 hilos, capa `pthreads`. `ej1.py` imprime esta información por stderr al iniciar.

## Matrices

Se descargan de SuiteSparse (formato Matrix Market) y se descomprimen en `matrices/<nombre>/<nombre>.mtx`.
Si sparse.tamu.edu responde 403, los mismos archivos están en `https://cise.ufl.edu/research/sparse/MM/<Grupo>/<nombre>.tar.gz`.

| Matriz | Grupo/Nombre |
|---|---|
| bcsstk17 | HB/bcsstk17 |
| gyro_m | Oberwolfach/gyro_m |
| msc10848 | Boeing/msc10848 |
| cfd1 | Rothberg/cfd1 |
| cant | Williams/cant |
| bmwcra_1 | GHS_psdef/bmwcra_1 |
| apache2 | GHS_psdef/apache2 |
| ecology2 | McRae/ecology2 |
| thermal2 | Schmid/thermal2 |

La primera lectura de cada matriz guarda una copia `matrices/<nombre>/<nombre>.npz` para acelerar las siguientes.

## Scripts

| Script | Qué hace |
|---|---|
| `comun.py` | Lectura de matrices, construcción de `b = A·1` y residuo relativo |
| `ej1.py` | Ejercicio 1: genera matrices densas con κ controlado y compara los métodos. Imprime un CSV por stdout |
| `ej2_directo.py` | Ejercicio 2 a) y b): factorización directa con distintos órdenes; cada caso corre en un subproceso |
| `ej2_cg.py` | Ejercicio 2 c): CG con los tres precondicionadores |
| `figs_ej1.py`, `figs_ej2.py` | Figuras a partir de los CSV |
| `tabla_ej2.py` | Lectura y unión de los CSV del Ejercicio 2 (la usan `figs_ej2.py` e `informe_ej2.py`) |
| `informe.py`, `informe_ej2.py` | Generan `informe_tarea2.docx` |

## Cómo regenerar resultados

```bash
./venv/bin/python ej1.py > ej1_resultados.csv
./venv/bin/python figs_ej1.py
./venv/bin/python ej2_directo.py > ej2_directo.csv 2> ej2_directo_err.txt
./venv/bin/python ej2_cg.py > ej2_cg.csv
./venv/bin/python figs_ej2.py
./venv/bin/python informe.py
```

Todos los scripts aceptan una lista de tamaños (`ej1.py 256,512`) o de matrices
(`ej2_directo.py gyro_m,cfd1`) para pruebas rápidas.
