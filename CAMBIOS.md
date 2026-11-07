# Cambios de medición y métricas (rama `mejoras-medicion`)

Todos los números de este documento salen de las corridas guardadas en `ej1_resultados.csv`, `ej2_directo.csv` y `ej2_cg.csv`.
Los resultados anteriores se conservan como `*_v1.csv`. Cada CSV nuevo agrega columnas al final; las anteriores no cambian de nombre ni de orden,
salvo `t_orden` de `ej2_directo.csv`, que se separó (ver más abajo).

## 1. Cambios por archivo

- `README.md` (nuevo): scripts, comandos para regenerar cada CSV y figura, fuente de cada matriz y hardware/BLAS.
- `requirements.txt` (nuevo): versiones instaladas de numpy, scipy (≥ 1.12), numba, pandas, matplotlib, python-docx y threadpoolctl.
- `ej1.py`, `ej2_directo.py`, `ej2_cg.py`: ejecución bajo `if __name__ == "__main__"`.
- `ej1.py`:
  - imprime `threadpool_info()` por stderr al iniciar;
  - `t_fact` y `t_solve` son la mediana de 3 repeticiones; las métricas de error se calculan una vez;
  - `lu_pivoteo_parcial` cuenta los intercambios de filas;
  - se calcula ρ = max|U| / max|A| para las dos LU propias;
  - un pivote exactamente 0 lanza `ZeroDivisionError` y la fila sale como `fallo:`;
  - columnas nuevas `n_swaps` y `rho` (vacías para los demás métodos);
  - método nuevo "QR (sin formar Q)": `qr(mode="raw")` y `dormqr` para aplicar Qᵀb. En cada tamaño con κ = 1e2 la diferencia relativa con la solución de QR salió entre 5.7e-15 y 1.2e-14 (se imprime por stderr), muy por debajo del umbral de 1e-10.
- `ej2_directo.py`:
  - `t_orden` se separó en `t_prep` (conversión a CSC, sin ordenar), `t_orden` (solo `reverse_cuthill_mckee` o el análisis de CHOLMOD) y `t_perm` (construir `A[p][:, p]`). El tiempo de MMD queda incluido en `t_fact`, porque lo calcula SuperLU dentro de `splu`;
  - columnas nuevas `banda`, `env` (para natural, rcm y amd_cholmod; vacías para MMD) y `estado` (`ok`, `timeout`, `fallo`). La banda y la envolvente se calculan antes de factorizar y el subproceso las imprime enseguida, de modo que las filas con timeout también las traen y se conservan `n` y `nnz`;
  - `ru_maxrss` se convierte según `sys.platform` (bytes en macOS, KB en Linux);
  - el stderr de los subprocesos que fallan o expiran va al stderr del script (`ej2_directo_err.txt` con la redirección del README);
  - orden opcional `amd_cholmod` (ver sección 4);
  - se mantiene el límite de 300 s por caso.
- `ej2_cg.py`:
  - pasada de calentamiento de Numba sobre una matriz SPD de 50×50 antes del bucle;
  - columnas nuevas `err_rel`, `reps`, `kappa_est`, `lmin`, `lmax`;
  - se repite el CG 3 veces (mediana de `t_setup` y de `t_cg`) solo si una primera pasada de la matriz suma menos de 10 s;
  - λmax con `eigsh(A, 1, which="LA")` y λmin con `eigsh(A, 1, sigma=0, which="LM")`, cada uno en un subproceso con límite de 120 s (NaN si lo excede o falla). Para λmin se pasa `OPinv` con la factorización LU simétrica con MMD que ya usa `ej2_directo.py`, en lugar de dejar que `eigsh` factorice con el orden por defecto de SuperLU; el resultado matemático es el mismo.
- `tabla_ej2.py`: lee el nuevo `ej2_directo.csv` (un solo archivo, con columna `estado`).
- `figs_ej1.py`: se agrega "QR (sin formar Q)" y la figura de tiempo contra n usa κ = 1e2.
- `figs_ej2.py`: sin cambios de código; se regeneraron las figuras con los CSV nuevos.

## 2. Comparación v1 contra resultados nuevos

Las matrices del Ejercicio 1 son las mismas (semilla fija) y los errores y residuos coinciden con v1 en todas las filas comunes; solo cambian los tiempos.

### 2.1 Setup de ILU(0) (compilación de Numba)

| Matriz | t_setup ILU(0) v1 (s) | t_setup ILU(0) nuevo (s) |
|---|---|---|
| gyro_m | 0.134 | 0.007401 |
| msc10848 | 0.00937 | 0.008743 |
| bcsstk17 | 0.000879 | 0.0006333 |
| cfd1 | 0.03003 | 0.02833 |
| cant | 0.04774 | 0.04825 |
| bmwcra_1 | 0.339 | 0.3463 |
| apache2 | 0.02428 | 0.02098 |
| ecology2 | 0.02333 | 0.02217 |
| thermal2 | 0.1478 | 0.1443 |

La diferencia se concentra en `gyro_m`, la primera matriz: de 0.134 s a 0.0074 s. En las demás matrices los valores son equivalentes a v1.

### 2.2 Tiempo de ordenamiento y de factorización, y memoria (Ej. 2, solver directo)

En v1 `t_orden` valía solo la conversión a CSC para natural y MMD, y la suma de RCM más la permutación para RCM. En la columna "Nuevo" se muestran las columnas separadas.
En las filas con timeout no hay `t_fact` ni RSS.

| Matriz | Orden | t_orden v1 (s) | Nuevo (s) | t_fact v1 (s) | t_fact nuevo (s) | RSS v1 (MB) | RSS nuevo (MB) |
|---|---|---|---|---|---|---|---|
| gyro_m | natural | 0.001058 | t_prep 0.001006 | 1.253 | 1.086 | 259 | 258.9 |
| gyro_m | rcm | 0.009935 | t_orden 0.002261 + t_perm 0.003134 | 0.08615 | 0.07298 | 117.1 | 113.2 |
| gyro_m | amd | 0.001124 | t_prep 0.001055 | 0.04622 | 0.04562 | 87.58 | 83.72 |
| msc10848 | natural | 0.004017 | t_prep 0.003613 | 0.4923 | 0.5037 | 224.1 | 233.9 |
| msc10848 | rcm | 0.02175 | t_orden 0.002979 + t_perm 0.00816 | 1.057 | 1 | 336.4 | 333.4 |
| msc10848 | amd | 0.0118 | t_prep 0.005239 | 0.2398 | 0.2152 | 179.4 | 175.8 |
| bcsstk17 | natural | 0.002089 | t_prep 0.001219 | 0.1214 | 0.1506 | 147.3 | 134.5 |
| bcsstk17 | rcm | 0.004751 | t_orden 0.001644 + t_perm 0.005437 | 0.2888 | 0.2696 | 169.9 | 170.7 |
| bcsstk17 | amd | 0.001266 | t_prep 0.001155 | 0.1075 | 0.07887 | 113.6 | 106.2 |
| cfd1 | natural | 0.003649 | t_prep 0.005308 | 56.33 | 54.71 | 3341 | 4416 |
| cfd1 | rcm | 0.03666 | t_orden 0.007856 + t_perm 0.01639 | 83.9 | 83.34 | 3878 | 5543 |
| cfd1 | amd | 0.005077 | t_prep 0.005297 | 18.47 | 18.41 | 1865 | 1860 |
| cant | natural | 0.006892 | t_prep 0.01449 | 1.349 | 1.379 | 1011 | 948.7 |
| cant | rcm | 0.04306 | t_orden 0.01244 + t_perm 0.03461 | 1.516 | 1.503 | 968.5 | 966.5 |
| cant | amd | 0.01028 | t_prep 0.01076 | 4.246 | 4.238 | 1218 | 1217 |
| bmwcra_1 | natural | 0.01887 | t_prep 0.03198 | 226.6 | 233.1 | 7930 | 7243 |
| bmwcra_1 | rcm | 0.1003 | t_orden 0.03125 + t_perm 0.0743 | 112.1 | 114.3 | 8325 | 5862 |
| bmwcra_1 | amd | 0.02553 | t_prep 0.02534 | 38.49 | 40.03 | 4535 | 4531 |
| apache2 | natural | - | t_prep 0.01688 | - | - | - | - |
| apache2 | rcm | - | t_orden 0.05077 + t_perm 0.09662 | - | - | - | - |
| apache2 | amd | 0.01772 | t_prep 0.01912 | 89.77 | 93.51 | 8790 | 6382 |
| ecology2 | natural | - | t_prep 0.023 | - | - | - | - |
| ecology2 | rcm | - | t_orden 0.05399 + t_perm 0.1239 | - | - | - | - |
| ecology2 | amd | 0.01573 | t_prep 0.02657 | 5.419 | 5.893 | 1985 | 1983 |
| thermal2 | natural | - | t_prep 0.03562 | - | - | - | - |
| thermal2 | rcm | - | t_orden 0.1617 + t_perm 0.1652 | - | - | - | - |
| thermal2 | amd | 0.03229 | t_prep 0.03245 | 52.9 | 54.57 | 4450 | 4448 |

Las corridas se hicieron una sola vez. Las memorias pico de `cfd1` con orden natural y RCM subieron respecto de v1 (3341 a 4416 MB y 3878 a 5543 MB), y la de RCM en `bmwcra_1` bajó (8325 a 5862 MB), mientras que con MMD coinciden. No investigué el motivo; conviene tomar el RSS de natural y RCM como orden de magnitud.

### 2.3 QR con y sin formar Q (n = 2048)

Tiempos en segundos.

| Tipo | κ | QR t_fact v1 | QR t_fact nuevo | QR sin Q t_fact | QR t_solve v1 | QR t_solve nuevo | QR sin Q t_solve | err QR | err QR sin Q |
|---|---|---|---|---|---|---|---|---|---|
| spd | 1e+02 | 0.5345 | 0.4286 | 0.2171 | 0.00249 | 0.00263 | 0.0134 | 9.85e-15 | 8.307e-15 |
| spd | 1e+06 | 0.4165 | 0.4451 | 0.2169 | 0.0024 | 0.00274 | 0.011 | 3.532e-11 | 2.672e-11 |
| spd | 1e+10 | 0.3987 | 0.4707 | 0.2199 | 0.00242 | 0.00367 | 0.01067 | 2.692e-07 | 1.78e-07 |
| indef | 1e+02 | 0.4001 | 0.4247 | 0.2215 | 0.00241 | 0.00258 | 0.01118 | 1.545e-14 | 1.209e-14 |
| indef | 1e+06 | 0.4374 | 0.4236 | 0.2297 | 0.00242 | 0.00263 | 0.01104 | 4.116e-11 | 2.643e-11 |
| indef | 1e+10 | 0.4047 | 0.4043 | 0.2147 | 0.0024 | 0.0026 | 0.01326 | 2.309e-07 | 1.414e-07 |

Sin formar Q la factorización tarda aproximadamente la mitad (0.21 a 0.23 s frente a 0.40 a 0.47 s), pero la resolución es unas 4 veces más lenta (0.011 s frente a 0.0026 s), porque aplica los reflectores con `dormqr` en vez de un producto matriz-vector con Q ya formada. Los errores son del mismo orden.

### 2.4 Cociente entre LU propia con pivoteo parcial y LAPACK (n = 2048)

| Tipo | κ | LU parcial / LAPACK v1 | LU parcial / LAPACK nuevo | LAPACK t_fact v1 (s) | LAPACK t_fact nuevo (s) |
|---|---|---|---|---|---|
| spd | 1e+02 | 61.9 | 53.7 | 0.0591 | 0.0642 |
| spd | 1e+06 | 37.9 | 57.3 | 0.0893 | 0.0602 |
| spd | 1e+10 | 54.6 | 56.6 | 0.0618 | 0.0607 |
| indef | 1e+02 | 56.1 | 59.2 | 0.061 | 0.0588 |
| indef | 1e+06 | 55.1 | 61 | 0.0619 | 0.0571 |
| indef | 1e+10 | 58.4 | 57.1 | 0.0582 | 0.0611 |

El cociente está entre 53.7 y 61.0 en las seis combinaciones. En v1 `spd` con κ = 1e6 daba 37.9 porque el punto de LAPACK era 0.0893 s, frente a 0.0602 s ahora.

## 3. Valores nuevos para el informe

### 3.1 Ejercicio 1: intercambios de filas y factor de crecimiento

| Tipo | n | κ | n_swaps (parcial) | ρ sin pivoteo | ρ pivoteo parcial | err sin pivoteo | err pivoteo parcial |
|---|---|---|---|---|---|---|---|
| spd | 512 | 1e+02 | 0 | 0.9204 | 0.9204 | 5.744e-15 | 5.744e-15 |
| spd | 512 | 1e+06 | 0 | 0.9431 | 0.9431 | 1.766e-11 | 1.766e-11 |
| spd | 512 | 1e+10 | 0 | 0.6009 | 0.6009 | 8.033e-08 | 8.033e-08 |
| spd | 1024 | 1e+02 | 0 | 0.9564 | 0.9564 | 9.043e-15 | 9.043e-15 |
| spd | 1024 | 1e+06 | 0 | 0.7352 | 0.7352 | 2.342e-11 | 2.342e-11 |
| spd | 1024 | 1e+10 | 0 | 0.6834 | 0.6834 | 1.494e-07 | 1.494e-07 |
| spd | 2048 | 1e+02 | 0 | 0.9419 | 0.9419 | 1.209e-14 | 1.209e-14 |
| spd | 2048 | 1e+06 | 0 | 0.8704 | 0.8704 | 3.248e-11 | 3.248e-11 |
| spd | 2048 | 1e+10 | 0 | 0.7896 | 0.7896 | 1.991e-07 | 1.991e-07 |
| indef | 512 | 1e+02 | 507 | 737.3 | 7.789 | 3.134e-12 | 3.947e-14 |
| indef | 512 | 1e+06 | 504 | 368.6 | 1.828 | 7.957e-09 | 3.355e-11 |
| indef | 512 | 1e+10 | 502 | 240.8 | 1.577 | 2.375e-05 | 1.538e-07 |
| indef | 1024 | 1e+02 | 1015 | 1.439e+04 | 6.307 | 4.347e-10 | 8.221e-14 |
| indef | 1024 | 1e+06 | 1018 | 1756 | 2.42 | 3.927e-08 | 1.003e-10 |
| indef | 1024 | 1e+10 | 1020 | 89.17 | 1.415 | 7.837e-05 | 4.628e-07 |
| indef | 2048 | 1e+02 | 2044 | 6615 | 20.74 | 3.74e-11 | 2.022e-13 |
| indef | 2048 | 1e+06 | 2039 | 1386 | 3.123 | 1.031e-07 | 2.502e-10 |
| indef | 2048 | 1e+10 | 2040 | 896.2 | 2.275 | 0.0002658 | 8.882e-07 |

En matrices SPD el pivoteo parcial no hizo ningún intercambio en las 9 combinaciones, y ρ < 1 (resultados idénticos con y sin pivoteo). En matrices indefinidas hizo entre n−10 y n−1 intercambios y el ρ de LU sin pivoteo llegó a 1.4e4 (n = 1024, κ = 1e2) y a 6.6e3 (n = 2048, κ = 1e2), contra 20.7 como máximo con pivoteo parcial.

### 3.2 Ejercicio 2: banda y envolvente

Calculadas sobre la matriz permutada, antes de factorizar. Para MMD no se calculan (la permutación queda dentro de SuperLU).

| Matriz | Banda natural | Env natural | Estado nat. | Banda RCM | Env RCM | Estado RCM |
|---|---|---|---|---|---|---|
| gyro_m | 5 142 | 14 309 541 | ok | 567 | 2 201 676 | ok |
| msc10848 | 10 706 | 50 044 035 | ok | 2 168 | 6 910 563 | ok |
| bcsstk17 | 521 | 2 837 421 | ok | 763 | 3 806 736 | ok |
| cfd1 | 6 229 | 103 460 652 | ok | 3 465 | 131 735 999 | ok |
| cant | 275 | 16 778 583 | ok | 654 | 17 152 263 | ok |
| bmwcra_1 | 141 050 | 978 933 140 | ok | 5 036 | 452 381 217 | ok |
| apache2 | 65 837 | 1 717 106 409 | timeout | 4 516 | 2 140 669 639 | timeout |
| ecology2 | 1 000 | 998 998 001 | timeout | 1 000 | 667 164 181 | timeout |
| thermal2 | 1 226 000 | 53 657 335 080 | timeout | 842 | 694 298 367 | timeout |

### 3.3 Ejercicio 2: número de condición estimado

λmax de `apache2` y de `ecology2` excedió los 120 s, así que su κ queda sin estimar.

| Matriz | λmin | λmax | κ estimado |
|---|---|---|---|
| gyro_m | 7.539e-14 | 1.888e-07 | 2.504e+06 |
| msc10848 | 63.19 | 6.298e+11 | 9.966e+09 |
| bcsstk17 | 1 | 1.296e+10 | 1.296e+10 |
| cfd1 | 2.005e-05 | 6.803 | 3.393e+05 |
| cant | 7.458e-07 | 1.943e+04 | 2.605e+10 |
| bmwcra_1 | 0.08722 | 3.282e+07 | 3.763e+08 |
| apache2 | 0.02995 | - | - |
| ecology2 | 1.226e-06 | - | - |
| thermal2 | 1.608e-06 | 7.806 | 4.854e+06 |

### 3.4 Ejercicio 2: CG con error relativo

`res_rel` es ‖b − Ax‖/‖b‖ y `err_rel` es ‖x − 1‖/‖1‖. Los tiempos son la mediana de `reps` repeticiones.

| Matriz | Precond. | Iter. | t_cg (s) | Res. rel. | Err. rel. | reps |
|---|---|---|---|---|---|---|
| gyro_m | sin precond. | 8425 | 1.945 | 9.79e-13 | 7.18e-09 | 3 |
| gyro_m | Jacobi | 511 | 0.1211 | 9.88e-13 | 2.38e-10 | 3 |
| gyro_m | ILU0 | 145 | 0.08911 | 8.85e-13 | 1.51e-10 | 3 |
| msc10848 | sin precond. | 20000 | 19.55 | 1.49e-07 | 0.0856 | 1 |
| msc10848 | Jacobi | 5374 | 5.276 | 8.68e-13 | 5.93e-08 | 1 |
| msc10848 | ILU0 | falla | 0 | - | - | 1 |
| bcsstk17 | sin precond. | 20000 | 5.39 | 9.39e-09 | 4.15e-06 | 3 |
| bcsstk17 | Jacobi | 3141 | 0.8592 | 9.68e-13 | 1.41e-10 | 3 |
| bcsstk17 | ILU0 | falla | 0 | - | - | 3 |
| cfd1 | sin precond. | 2200 | 2.185 | 9.67e-13 | 3.8e-11 | 3 |
| cfd1 | Jacobi | 2200 | 2.238 | 9.67e-13 | 3.8e-11 | 3 |
| cfd1 | ILU0 | 504 | 1.509 | 9.56e-13 | 1.25e-11 | 3 |
| cant | sin precond. | 11328 | 32.25 | 9.74e-13 | 1.71e-07 | 1 |
| cant | Jacobi | 4576 | 12.97 | 1e-12 | 1.72e-07 | 1 |
| cant | ILU0 | falla | 0 | - | - | 1 |
| bmwcra_1 | sin precond. | 12295 | 95.14 | 1.01e-12 | 2.92e-11 | 1 |
| bmwcra_1 | Jacobi | 6259 | 48.83 | 1.01e-12 | 3.55e-11 | 1 |
| bmwcra_1 | ILU0 | 1475 | 27.32 | 9.54e-13 | 3.32e-11 | 1 |
| apache2 | sin precond. | 6018 | 23.43 | 9.51e-11 | 6.55e-13 | 1 |
| apache2 | Jacobi | 5017 | 20.98 | 8.75e-11 | 1.43e-13 | 1 |
| apache2 | ILU0 | 958 | 13.41 | 3.81e-11 | 2.44e-13 | 1 |
| ecology2 | sin precond. | 7485 | 36.71 | 7.46e-11 | 3.62e-12 | 1 |
| ecology2 | Jacobi | 7447 | 38.84 | 7.47e-11 | 2.97e-12 | 1 |
| ecology2 | ILU0 | 2224 | 42.29 | 4.05e-11 | 6.01e-12 | 1 |
| thermal2 | sin precond. | 6336 | 77.73 | 1.05e-12 | 2.78e-11 | 1 |
| thermal2 | Jacobi | 6052 | 77.11 | 1.04e-12 | 1.98e-11 | 1 |
| thermal2 | ILU0 | 2496 | 90.2 | 1.02e-12 | 3.02e-11 | 1 |

### 3.5 AMD de CHOLMOD

No hay resultados (ver sección 4).

## 4. Lo que no se pudo hacer

- **Orden `amd_cholmod`:** `scikit-sparse` no se pudo instalar. No hay Homebrew en el equipo (no se puede instalar SuiteSparse) y pip no encontró ninguna rueda binaria (`--only-binary`) para este equipo, con Python 3.9 en macOS ARM. El código está escrito y se activa solo si `sksparse` está disponible, pero no pude ejecutarlo, así que no hay ningún resultado de AMD real y la rama de código no está probada. Para activarlo: `brew install suite-sparse && pip install scikit-sparse`. Mientras tanto, el "AMD" del informe sigue siendo MMD de SuperLU.
- **κ de `apache2` y de `ecology2`:** λmax no terminó en 120 s. Hay λmin para ambas.
- **Factorizaciones con timeout:** `apache2`, `ecology2` y `thermal2` con orden natural y RCM no terminaron en 300 s. Las filas guardan banda y envolvente, pero no se sabe si el motivo fue solo el tiempo o también la memoria, porque el stderr de los subprocesos expirados quedó vacío.
- **Una sola ejecución por configuración** en el solver directo y, para las matrices cuyo CG suma 10 s o más, también en el CG. No hay medida de variabilidad entre corridas; los RSS de la sección 2.2 muestran que puede ser grande.
- **Informe:** no se modificaron `informe_tarea2.docx` ni `informe*.py`. Quedan por actualizar a mano: el pie de la Figura 3 (dice κ = 10⁶ y la figura ahora usa κ = 10²), las tablas con los valores nuevos y el cociente de ≈55× entre la LU propia y LAPACK.
- **Archivos ajenos:** `matrices2.py` y `AMD-2.6-py3-none-any.whl` están en la carpeta del proyecto y no se versionaron.
