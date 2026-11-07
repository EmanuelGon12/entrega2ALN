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
  - `t_orden` se separó en `t_prep` (conversión a CSC, sin ordenar), `t_orden` (solo `reverse_cuthill_mckee` o el AMD de SuiteSparse) y `t_perm` (construir `A[p][:, p]`). El tiempo de MMD queda incluido en `t_fact`, porque lo calcula SuperLU dentro de `splu`;
  - columnas nuevas `banda`, `env` (para natural, rcm y amd_real; vacías para MMD) y `estado` (`ok`, `timeout`, `fallo`). La banda y la envolvente se calculan antes de factorizar y el subproceso las imprime enseguida, de modo que las filas con timeout también las traen y se conservan `n` y `nnz`;
  - `ru_maxrss` se convierte según `sys.platform` (bytes en macOS, KB en Linux);
  - el stderr de los subprocesos que fallan o expiran va al stderr del script (`ej2_directo_err.txt` con la redirección del README);
  - orden `amd_real`: AMD de SuiteSparse vía `cvxopt.amd.order` (o `sksparse` si está instalado);
  - `REPS` repeticiones por caso (variable de entorno, 3 en las corridas); `t_fact`/`rss_MB` son la mediana y se agregan `reps`, `t_fact_min`, `t_fact_max`, `rss_min`, `rss_max`;
  - muestreo cada 0.5 s del RSS del subproceso y del swap del sistema (`psutil`): columnas `rss_obs_MB` y `swap_MB`;
  - se mantiene el límite de 300 s por caso.
- `ej2_cg.py`:
  - pasada de calentamiento de Numba sobre una matriz SPD de 50×50 antes del bucle;
  - columnas nuevas `err_rel`, `reps`, `kappa_est`, `lmin`, `lmax`;
  - se repite el CG 3 veces en todas las matrices (mediana de `t_setup` y de `t_cg`);
  - λmax con `eigsh(A, 1, which="LA")` y λmin con `eigsh(A, 1, sigma=0, which="LM")`, cada uno en un subproceso con límite de 300 s (λmax con `tol=1e-6`) (NaN si lo excede o falla). Para λmin se pasa `OPinv` con la factorización LU simétrica con MMD que ya usa `ej2_directo.py`, en lugar de dejar que `eigsh` factorice con el orden por defecto de SuperLU; el resultado matemático es el mismo.
- `tabla_ej2.py`: lee el nuevo `ej2_directo.csv` (un solo archivo, con columna `estado`).
- `figs_ej1.py`: se agrega "QR (sin formar Q)" y la figura de tiempo contra n usa κ = 1e2.
- `figs_ej2.py`: se agrega el orden `amd_real` ("AMD (SuiteSparse)") junto a MMD en las figuras de tiempo y memoria; se regeneraron con los CSV nuevos.
- `requirements.txt`: se agregan `cvxopt` y `psutil`.

## 2. Comparación v1 contra resultados nuevos

Las matrices del Ejercicio 1 son las mismas (semilla fija) y los errores y residuos coinciden con v1 en todas las filas comunes; solo cambian los tiempos.

### 2.1 Setup de ILU(0) (compilación de Numba)

| Matriz | t_setup ILU(0) v1 (s) | t_setup ILU(0) nuevo (s) |
|---|---|---|
| gyro_m | 0.134 | 0.007275 |
| msc10848 | 0.00937 | 0.00862 |
| bcsstk17 | 0.000879 | 0.0006427 |
| cfd1 | 0.03003 | 0.02838 |
| cant | 0.04774 | 0.04577 |
| bmwcra_1 | 0.339 | 0.3366 |
| apache2 | 0.02428 | 0.02171 |
| ecology2 | 0.02333 | 0.02105 |
| thermal2 | 0.1478 | 0.1439 |

La diferencia se concentra en `gyro_m`, la primera matriz: de 0.134 s a 0.0074 s. En las demás matrices los valores son equivalentes a v1.

### 2.2 Tiempo de factorización y memoria (Ej. 2, solver directo)

En v1 `t_orden` valía solo la conversión a CSC para natural y MMD, y la suma de RCM más la permutación para RCM; en el CSV nuevo se separó en `t_prep`, `t_orden` y `t_perm` (ver sección 1). Aquí se comparan `t_fact` y el RSS. En las filas con timeout no hay valores.

| Matriz | Orden | t_orden v1 (s) | t_fact v1 (s) | t_fact nuevo (s) | RSS v1 (MB) | RSS nuevo (MB) |
|---|---|---|---|---|---|---|
| gyro_m | natural | 0.001058 | 1.253 | 1.164 | 259 | 261.5 |
| gyro_m | rcm | 0.009935 | 0.08615 | 0.0723 | 117.1 | 116.4 |
| gyro_m | amd | 0.001124 | 0.04622 | 0.03909 | 87.58 | 83.81 |
| msc10848 | natural | 0.004017 | 0.4923 | 0.4949 | 224.1 | 235.8 |
| msc10848 | rcm | 0.02175 | 1.057 | 1.024 | 336.4 | 333.4 |
| msc10848 | amd | 0.0118 | 0.2398 | 0.2209 | 179.4 | 178.1 |
| bcsstk17 | natural | 0.002089 | 0.1214 | 0.1734 | 147.3 | 135.3 |
| bcsstk17 | rcm | 0.004751 | 0.2888 | 0.2767 | 169.9 | 167.8 |
| bcsstk17 | amd | 0.001266 | 0.1075 | 0.08986 | 113.6 | 108.3 |
| cfd1 | natural | 0.003649 | 56.33 | 55.54 | 3341 | 4417 |
| cfd1 | rcm | 0.03666 | 83.9 | 86.29 | 3878 | 5302 |
| cfd1 | amd | 0.005077 | 18.47 | 19.81 | 1865 | 1862 |
| cant | natural | 0.006892 | 1.349 | 1.409 | 1011 | 949.4 |
| cant | rcm | 0.04306 | 1.516 | 1.517 | 968.5 | 967.5 |
| cant | amd | 0.01028 | 4.246 | 4.422 | 1218 | 1219 |
| bmwcra_1 | natural | 0.01887 | 226.6 | 231.6 | 7930 | 8528 |
| bmwcra_1 | rcm | 0.1003 | 112.1 | 113 | 8325 | 8466 |
| bmwcra_1 | amd | 0.02553 | 38.49 | 39.48 | 4535 | 4533 |
| apache2 | natural | - | - | - | - | - |
| apache2 | rcm | - | - | - | - | - |
| apache2 | amd | 0.01772 | 89.77 | 89.93 | 8790 | 9273 |
| ecology2 | natural | - | - | - | - | - |
| ecology2 | rcm | - | - | - | - | - |
| ecology2 | amd | 0.01573 | 5.419 | 5.364 | 1985 | 1984 |
| thermal2 | natural | - | - | - | - | - |
| thermal2 | rcm | - | - | - | - | - |
| thermal2 | amd | 0.03229 | 52.9 | 52.42 | 4450 | 4449 |

**Variabilidad entre corridas.** Cada caso se repitió 3 veces (mediana en `t_fact` y `rss_MB`; mínimo y máximo en `t_fact_min`, `t_fact_max`, `rss_min`, `rss_max`). El tiempo de factorización varió como máximo 38.3 % de la mediana (gyro_m, amd_real); en 22 de los 30 casos exitosos fue menor que 5 %; los mayores (hasta 38 %) son factorizaciones de menos de 0.3 s. La memoria pico fue estable salvo en los casos con órdenes que generan mucho relleno: cfd1 natural 4057 a 4420 MB (mediana 4417), cfd1 rcm 5225 a 5544 MB (mediana 5302), bmwcra_1 natural 8198 a 8690 MB (mediana 8528), bmwcra_1 rcm 8352 a 9271 MB (mediana 8466). Esto es coherente con las diferencias de RSS entre v1 y la corrida anterior (sección 2.2), pero no se identificó la causa; `bmwcra_1` natural y RCM usaron 6789 y 543 MB de swap aun terminando, lo que apunta a presión de memoria del sistema.

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

Calculadas sobre la matriz permutada, antes de factorizar. Para MMD no se calculan (la permutación queda dentro de SuperLU). La banda y la envolvente del orden AMD de SuiteSparse son enormes (del orden de n) porque AMD no busca reducirlas; sirven solo como comparación.

| Matriz | Banda nat. | Env nat. | Estado nat. | Banda RCM | Env RCM | Estado RCM | Banda AMD | Env AMD | Estado AMD |
|---|---|---|---|---|---|---|---|---|---|
| gyro_m | 5 142 | 14 309 541 | ok | 567 | 2 201 676 | ok | 5 785 | 4 122 216 | ok |
| msc10848 | 10 706 | 50 044 035 | ok | 2 168 | 6 910 563 | ok | 10 813 | 10 542 900 | ok |
| bcsstk17 | 521 | 2 837 421 | ok | 763 | 3 806 736 | ok | 10 418 | 12 331 725 | ok |
| cfd1 | 6 229 | 103 460 652 | ok | 3 465 | 131 735 999 | ok | 70 238 | 379 829 389 | ok |
| cant | 275 | 16 778 583 | ok | 654 | 17 152 263 | ok | 61 375 | 372 793 830 | ok |
| bmwcra_1 | 141 050 | 978 933 140 | ok | 5 036 | 452 381 217 | ok | 148 696 | 1 316 816 035 | ok |
| apache2 | 65 837 | 1 717 106 409 | timeout | 4 516 | 2 140 669 639 | timeout | 711 651 | 6 179 539 286 | ok |
| ecology2 | 1 000 | 998 998 001 | timeout | 1 000 | 667 164 181 | timeout | 998 543 | 4 007 927 715 | ok |
| thermal2 | 1 226 000 | 53 657 335 080 | timeout | 842 | 694 298 367 | timeout | 1 227 011 | 6 344 948 523 | ok |

### 3.3 Ejercicio 2: número de condición estimado

λmax con `eigsh(which="LA", tol=1e-6)` (límite de 300 s) y λmin con inversión desplazada.

| Matriz | λmin | λmax | κ estimado |
|---|---|---|---|
| gyro_m | 7.539e-14 | 1.888e-07 | 2.504e+06 |
| msc10848 | 63.19 | 6.298e+11 | 9.966e+09 |
| bcsstk17 | 1 | 1.296e+10 | 1.296e+10 |
| cfd1 | 2.005e-05 | 6.803 | 3.393e+05 |
| cant | 7.458e-07 | 1.943e+04 | 2.605e+10 |
| bmwcra_1 | 0.08722 | 3.282e+07 | 3.763e+08 |
| apache2 | 0.02995 | 9.187e+04 | 3.068e+06 |
| ecology2 | 1.226e-06 | 80 | 6.527e+07 |
| thermal2 | 1.608e-06 | 7.806 | 4.854e+06 |

### 3.4 Ejercicio 2: CG con error relativo

`res_rel` es ‖b − Ax‖/‖b‖ y `err_rel` es ‖x − 1‖/‖1‖. Los tiempos son la mediana de `reps` repeticiones (3 en todos los casos de esta corrida).

| Matriz | Precond. | Iter. | t_cg (s) | Res. rel. | Err. rel. | reps |
|---|---|---|---|---|---|---|
| gyro_m | sin precond. | 8425 | 1.946 | 9.79e-13 | 7.18e-09 | 3 |
| gyro_m | Jacobi | 511 | 0.121 | 9.88e-13 | 2.38e-10 | 3 |
| gyro_m | ILU0 | 145 | 0.08906 | 8.85e-13 | 1.51e-10 | 3 |
| msc10848 | sin precond. | 20000 | 19.55 | 1.49e-07 | 0.0856 | 3 |
| msc10848 | Jacobi | 5374 | 5.271 | 8.68e-13 | 5.93e-08 | 3 |
| msc10848 | ILU0 | falla | 0 | - | - | 3 |
| bcsstk17 | sin precond. | 20000 | 5.388 | 9.39e-09 | 4.15e-06 | 3 |
| bcsstk17 | Jacobi | 3141 | 0.8575 | 9.68e-13 | 1.41e-10 | 3 |
| bcsstk17 | ILU0 | falla | 0 | - | - | 3 |
| cfd1 | sin precond. | 2200 | 2.188 | 9.67e-13 | 3.8e-11 | 3 |
| cfd1 | Jacobi | 2200 | 2.242 | 9.67e-13 | 3.8e-11 | 3 |
| cfd1 | ILU0 | 504 | 1.508 | 9.56e-13 | 1.25e-11 | 3 |
| cant | sin precond. | 11328 | 31.66 | 9.74e-13 | 1.71e-07 | 3 |
| cant | Jacobi | 4576 | 12.89 | 1e-12 | 1.72e-07 | 3 |
| cant | ILU0 | falla | 0 | - | - | 3 |
| bmwcra_1 | sin precond. | 12295 | 92.73 | 1.01e-12 | 2.92e-11 | 3 |
| bmwcra_1 | Jacobi | 6259 | 47.5 | 1.01e-12 | 3.55e-11 | 3 |
| bmwcra_1 | ILU0 | 1475 | 26.99 | 9.54e-13 | 3.32e-11 | 3 |
| apache2 | sin precond. | 6018 | 22.91 | 9.51e-11 | 6.55e-13 | 3 |
| apache2 | Jacobi | 5017 | 20.3 | 8.75e-11 | 1.43e-13 | 3 |
| apache2 | ILU0 | 958 | 13.32 | 3.81e-11 | 2.44e-13 | 3 |
| ecology2 | sin precond. | 7485 | 34.94 | 7.46e-11 | 3.62e-12 | 3 |
| ecology2 | Jacobi | 7447 | 37.83 | 7.47e-11 | 2.97e-12 | 3 |
| ecology2 | ILU0 | 2224 | 41.5 | 4.05e-11 | 6.01e-12 | 3 |
| thermal2 | sin precond. | 6336 | 76.65 | 1.05e-12 | 2.78e-11 | 3 |
| thermal2 | Jacobi | 6052 | 76.47 | 1.04e-12 | 1.98e-11 | 3 |
| thermal2 | ILU0 | 2496 | 89.84 | 1.02e-12 | 3.02e-11 | 3 |

### 3.5 AMD de SuiteSparse (`amd_real`)

Orden calculado con `cvxopt.amd.order` sobre el triángulo inferior de A; luego `A[p][:, p]` y `splu` con orden natural. "t total" suma ordenamiento, permutación y factorización; en MMD el ordenamiento ya está dentro de `t_fact`.

| Matriz | nnz(L+U) MMD | nnz(L+U) AMD | t total MMD (s) | t total AMD (s) | RSS MMD (MB) | RSS AMD (MB) |
|---|---|---|---|---|---|---|
| gyro_m | 8.302e+05 | 8.332e+05 | 0.03909 | 0.07209 | 83.81 | 95.98 |
| msc10848 | 4.118e+06 | 3.933e+06 | 0.2209 | 0.277 | 178.1 | 220.2 |
| bcsstk17 | 2.012e+06 | 2.087e+06 | 0.08986 | 0.1091 | 108.3 | 127.9 |
| cfd1 | 7.566e+07 | 7.521e+07 | 19.81 | 11.72 | 1862 | 1796 |
| cant | 4.857e+07 | 5.789e+07 | 4.422 | 5.843 | 1219 | 1592 |
| bmwcra_1 | 1.921e+08 | 1.944e+08 | 39.48 | 33.58 | 4533 | 4996 |
| apache2 | 3.845e+08 | 3.546e+08 | 89.93 | 71.75 | 9273 | 8999 |
| ecology2 | 7.703e+07 | 9.151e+07 | 5.364 | 6.134 | 1984 | 2926 |
| thermal2 | 1.31e+08 | 1.298e+08 | 52.42 | 9.264 | 4449 | 4211 |

AMD real da menos relleno que MMD en 4 de las 9 matrices (`msc10848`, `cfd1`, `apache2`, `thermal2`) y más en las otras 5 (`gyro_m`, `bcsstk17`, `cant`, `bmwcra_1`, `ecology2`), con diferencias de hasta 19 % (`cant`: 4.86e7 contra 5.79e7). En tiempo total gana en `cfd1`, `bmwcra_1`, `apache2` y `thermal2` (9.3 s contra 52.4 s, donde la mayor parte del tiempo de MMD parece estar en el propio ordenamiento); pierde en el resto.

**Variabilidad entre corridas.** Cada caso se repitió 3 veces (mediana en `t_fact` y `rss_MB`; mínimo y máximo en `t_fact_min`, `t_fact_max`, `rss_min`, `rss_max`). El tiempo de factorización varió como máximo 38.3 % de la mediana (gyro_m, amd_real); en 22 de los 30 casos exitosos fue menor que 5 %; los mayores (hasta 38 %) son factorizaciones de menos de 0.3 s. La memoria pico fue estable salvo en los casos con órdenes que generan mucho relleno: cfd1 natural 4057 a 4420 MB (mediana 4417), cfd1 rcm 5225 a 5544 MB (mediana 5302), bmwcra_1 natural 8198 a 8690 MB (mediana 8528), bmwcra_1 rcm 8352 a 9271 MB (mediana 8466). Esto es coherente con las diferencias de RSS entre v1 y la corrida anterior (sección 2.2), pero no se identificó la causa; `bmwcra_1` natural y RCM usaron 6789 y 543 MB de swap aun terminando, lo que apunta a presión de memoria del sistema.

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

Calculadas sobre la matriz permutada, antes de factorizar. Para MMD no se calculan (la permutación queda dentro de SuperLU). La banda y la envolvente del orden AMD de SuiteSparse son enormes (del orden de n) porque AMD no busca reducirlas; sirven solo como comparación.

| Matriz | Banda nat. | Env nat. | Estado nat. | Banda RCM | Env RCM | Estado RCM | Banda AMD | Env AMD | Estado AMD |
|---|---|---|---|---|---|---|---|---|---|
| gyro_m | 5 142 | 14 309 541 | ok | 567 | 2 201 676 | ok | 5 785 | 4 122 216 | ok |
| msc10848 | 10 706 | 50 044 035 | ok | 2 168 | 6 910 563 | ok | 10 813 | 10 542 900 | ok |
| bcsstk17 | 521 | 2 837 421 | ok | 763 | 3 806 736 | ok | 10 418 | 12 331 725 | ok |
| cfd1 | 6 229 | 103 460 652 | ok | 3 465 | 131 735 999 | ok | 70 238 | 379 829 389 | ok |
| cant | 275 | 16 778 583 | ok | 654 | 17 152 263 | ok | 61 375 | 372 793 830 | ok |
| bmwcra_1 | 141 050 | 978 933 140 | ok | 5 036 | 452 381 217 | ok | 148 696 | 1 316 816 035 | ok |
| apache2 | 65 837 | 1 717 106 409 | timeout | 4 516 | 2 140 669 639 | timeout | 711 651 | 6 179 539 286 | ok |
| ecology2 | 1 000 | 998 998 001 | timeout | 1 000 | 667 164 181 | timeout | 998 543 | 4 007 927 715 | ok |
| thermal2 | 1 226 000 | 53 657 335 080 | timeout | 842 | 694 298 367 | timeout | 1 227 011 | 6 344 948 523 | ok |

### 3.3 Ejercicio 2: número de condición estimado

λmax con `eigsh(which="LA", tol=1e-6)` (límite de 300 s) y λmin con inversión desplazada.

| Matriz | λmin | λmax | κ estimado |
|---|---|---|---|
| gyro_m | 7.539e-14 | 1.888e-07 | 2.504e+06 |
| msc10848 | 63.19 | 6.298e+11 | 9.966e+09 |
| bcsstk17 | 1 | 1.296e+10 | 1.296e+10 |
| cfd1 | 2.005e-05 | 6.803 | 3.393e+05 |
| cant | 7.458e-07 | 1.943e+04 | 2.605e+10 |
| bmwcra_1 | 0.08722 | 3.282e+07 | 3.763e+08 |
| apache2 | 0.02995 | 9.187e+04 | 3.068e+06 |
| ecology2 | 1.226e-06 | 80 | 6.527e+07 |
| thermal2 | 1.608e-06 | 7.806 | 4.854e+06 |

### 3.4 Ejercicio 2: CG con error relativo

`res_rel` es ‖b − Ax‖/‖b‖ y `err_rel` es ‖x − 1‖/‖1‖. Los tiempos son la mediana de `reps` repeticiones (3 en todos los casos de esta corrida).

| Matriz | Precond. | Iter. | t_cg (s) | Res. rel. | Err. rel. | reps |
|---|---|---|---|---|---|---|
| gyro_m | sin precond. | 8425 | 1.946 | 9.79e-13 | 7.18e-09 | 3 |
| gyro_m | Jacobi | 511 | 0.121 | 9.88e-13 | 2.38e-10 | 3 |
| gyro_m | ILU0 | 145 | 0.08906 | 8.85e-13 | 1.51e-10 | 3 |
| msc10848 | sin precond. | 20000 | 19.55 | 1.49e-07 | 0.0856 | 3 |
| msc10848 | Jacobi | 5374 | 5.271 | 8.68e-13 | 5.93e-08 | 3 |
| msc10848 | ILU0 | falla | 0 | - | - | 3 |
| bcsstk17 | sin precond. | 20000 | 5.388 | 9.39e-09 | 4.15e-06 | 3 |
| bcsstk17 | Jacobi | 3141 | 0.8575 | 9.68e-13 | 1.41e-10 | 3 |
| bcsstk17 | ILU0 | falla | 0 | - | - | 3 |
| cfd1 | sin precond. | 2200 | 2.188 | 9.67e-13 | 3.8e-11 | 3 |
| cfd1 | Jacobi | 2200 | 2.242 | 9.67e-13 | 3.8e-11 | 3 |
| cfd1 | ILU0 | 504 | 1.508 | 9.56e-13 | 1.25e-11 | 3 |
| cant | sin precond. | 11328 | 31.66 | 9.74e-13 | 1.71e-07 | 3 |
| cant | Jacobi | 4576 | 12.89 | 1e-12 | 1.72e-07 | 3 |
| cant | ILU0 | falla | 0 | - | - | 3 |
| bmwcra_1 | sin precond. | 12295 | 92.73 | 1.01e-12 | 2.92e-11 | 3 |
| bmwcra_1 | Jacobi | 6259 | 47.5 | 1.01e-12 | 3.55e-11 | 3 |
| bmwcra_1 | ILU0 | 1475 | 26.99 | 9.54e-13 | 3.32e-11 | 3 |
| apache2 | sin precond. | 6018 | 22.91 | 9.51e-11 | 6.55e-13 | 3 |
| apache2 | Jacobi | 5017 | 20.3 | 8.75e-11 | 1.43e-13 | 3 |
| apache2 | ILU0 | 958 | 13.32 | 3.81e-11 | 2.44e-13 | 3 |
| ecology2 | sin precond. | 7485 | 34.94 | 7.46e-11 | 3.62e-12 | 3 |
| ecology2 | Jacobi | 7447 | 37.83 | 7.47e-11 | 2.97e-12 | 3 |
| ecology2 | ILU0 | 2224 | 41.5 | 4.05e-11 | 6.01e-12 | 3 |
| thermal2 | sin precond. | 6336 | 76.65 | 1.05e-12 | 2.78e-11 | 3 |
| thermal2 | Jacobi | 6052 | 76.47 | 1.04e-12 | 1.98e-11 | 3 |
| thermal2 | ILU0 | 2496 | 89.84 | 1.02e-12 | 3.02e-11 | 3 |

### 3.5 AMD de SuiteSparse (`amd_real`)

Orden calculado con `cvxopt.amd.order` sobre el triángulo inferior de A; luego `A[p][:, p]` y `splu` con orden natural. "t total" suma ordenamiento, permutación y factorización; en MMD el ordenamiento ya está dentro de `t_fact`.

| Matriz | nnz(L+U) MMD | nnz(L+U) AMD | t total MMD (s) | t total AMD (s) | RSS MMD (MB) | RSS AMD (MB) |
|---|---|---|---|---|---|---|
| gyro_m | 8.302e+05 | 8.332e+05 | 0.03909 | 0.07209 | 83.81 | 95.98 |
| msc10848 | 4.118e+06 | 3.933e+06 | 0.2209 | 0.277 | 178.1 | 220.2 |
| bcsstk17 | 2.012e+06 | 2.087e+06 | 0.08986 | 0.1091 | 108.3 | 127.9 |
| cfd1 | 7.566e+07 | 7.521e+07 | 19.81 | 11.72 | 1862 | 1796 |
| cant | 4.857e+07 | 5.789e+07 | 4.422 | 5.843 | 1219 | 1592 |
| bmwcra_1 | 1.921e+08 | 1.944e+08 | 39.48 | 33.58 | 4533 | 4996 |
| apache2 | 3.845e+08 | 3.546e+08 | 89.93 | 71.75 | 9273 | 8999 |
| ecology2 | 7.703e+07 | 9.151e+07 | 5.364 | 6.134 | 1984 | 2926 |
| thermal2 | 1.31e+08 | 1.298e+08 | 52.42 | 9.264 | 4449 | 4211 |

AMD real da menos relleno que MMD en 4 de las 9 matrices. Las dos matrices donde da más relleno son `bcsstk17`/`cant`/`gyro_m`-tipo según la tabla; ver filas con nnz(L+U) AMD mayor que MMD.

## 4. Lo que no se pudo hacer o quedó abierto

- **`scikit-sparse`:** sigue sin poder instalarse (no hay Homebrew ni rueda binaria). Se resolvió con `cvxopt`, que incluye el AMD de SuiteSparse; el orden se llama `amd_real`. La rama de `sksparse` del código no se ejecutó.
- **Causa exacta de los timeouts (300 s):** `ej2_directo.py` ahora muestrea cada 0.5 s la memoria residente del subproceso y el swap del sistema antes de matarlo. Resultado en los 6 casos con timeout:
  - apache2, natural: RSS observado 7931 MB, swap usado 0 MB
  - apache2, rcm: RSS observado 8862 MB, swap usado 0 MB
  - ecology2, natural: RSS observado 8937 MB, swap usado 1188 MB
  - ecology2, rcm: RSS observado 8745 MB, swap usado 3818 MB
  - thermal2, natural: RSS observado 8157 MB, swap usado 0 MB
  - thermal2, rcm: RSS observado 9509 MB, swap usado 16720 MB

  Con esto se sabe que la memoria estaba en 8 a 9.5 GB (de 16 GB) en todos, y que en `ecology2` (natural y RCM) y `thermal2` (RCM) el sistema ya usaba swap; en `thermal2` con RCM, 16.7 GB. En esos tres casos el tiempo se consumió con el sistema paginando. En `apache2` (natural y RCM) y `thermal2` natural no hubo swap, así que lo que se mide es solo que no terminaron en 300 s con 8 a 9 GB; no se puede decir cuánto más habrían tardado ni si habrían terminado.
- **Valores del RSS observado:** `rss_obs_MB` es el máximo de un muestreo cada 0.5 s, por lo que subestima el pico en casos muy cortos (valores de 0.7 MB son de procesos que terminaron antes de la primera muestra). Para esos se usa `rss_MB` (`ru_maxrss`).
- **Informe:** no se modificaron `informe_tarea2.docx` ni `informe*.py`. Quedan por actualizar a mano: el pie de la Figura 3 (dice κ = 10⁶ y la figura ahora usa κ = 10²), las tablas con los valores nuevos, el cociente de ≈55× entre la LU propia y LAPACK y el texto que llama "AMD" a MMD de SuperLU (ahora hay un AMD real).
- **Archivos ajenos:** `matrices2.py` y `AMD-2.6-py3-none-any.whl` están en la carpeta del proyecto y no se versionaron.
