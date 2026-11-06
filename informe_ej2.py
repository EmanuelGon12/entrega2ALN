from comun import MATRICES
from tabla_ej2 import directo, cg

dd = directo()
cc = cg()
NOMBRE_ORD = {"natural": "Natural", "rcm": "RCM", "amd": "MMD"}


def tabla_generica(cab, filas, tam=8.5):
    t = doc.add_table(rows=1, cols=len(cab))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(cab):
        t.rows[0].cells[i].text = c
    for f in filas:
        cel = t.add_row().cells
        for i, v in enumerate(f):
            cel[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(tam)


def leyenda(txt):
    doc.add_paragraph(txt).runs[0].italic = True


doc.add_heading("Ejercicio 2", 1)
doc.add_heading("Matrices y entorno de trabajo", 2)
par("Se emplearon nueve matrices simétricas definidas positivas de la colección SuiteSparse (grupos HB, Oberwolfach, Boeing, "
    "Rothberg, Williams, GHS_psdef, McRae y Schmid), con órdenes entre 1.1·10⁴ y 1.2·10⁶ y entre 5 y 113 no nulos por fila, "
    "provenientes de problemas de elasticidad estructural, mecánica de fluidos, transferencia de calor y ecología. "
    "La Tabla 3 resume sus características. En todos los casos se tomó b = A·1, de modo que la solución exacta es el vector de unos. Todos los experimentos se ejecutaron secuencialmente en una única máquina de 16 GB de memoria, con una única ejecución por configuración.")
filas = []
for m in MATRICES:
    f = dd[(dd.matriz == m) & (dd.estado == "ok")].iloc[0]
    filas.append([m, f"{int(f.n):,}".replace(",", " "), f"{int(f.nnz):,}".replace(",", " "), f"{f.nnz / f.n:.1f}"])
leyenda("Tabla 3. Matrices utilizadas (nnz cuenta ambos triángulos).")
tabla_generica(["Matriz", "n", "nnz", "nnz/n"], filas, 9)
doc.add_paragraph()

doc.add_heading("Parte A – Solver directo: tiempo y memoria", 1)
doc.add_heading("Metodología", 2)
par("El solver directo es SuperLU a través de scipy.sparse.linalg.splu, configurado en modo simétrico y sin pivoteo por umbral "
    "(diag_pivot_thresh = 0), lo que lo hace equivalente a una factorización LU simétrica sobre la estructura de A. Esta elección "
    "se debió a que no fue posible instalar CHOLMOD en el entorno de trabajo. Por ello, el relleno se reporta como nnz(L) + nnz(U), "
    "aproximadamente el doble del factor de Cholesky. Cada combinación (matriz, orden) "
    "se ejecutó en un proceso independiente, con un límite de 300 s, para poder medir la memoria pico residente (RSS) sin contaminación "
    "entre casos. Se midieron el tiempo de reordenamiento, el de factorización y el de resolución, el relleno, la memoria pico, "
    "el residuo relativo ‖b − Ax‖/‖b‖ y el error relativo.")

doc.add_heading("Resultados", 2)
par("La Tabla 4 muestra la resolución con el orden natural de las matrices. Una marca «>300 s» indica que la factorización no terminó dentro del límite establecido.")
filas = []
for m in MATRICES:
    f = dd[(dd.matriz == m) & (dd.orden == "natural")].iloc[0]
    if f.estado == "ok":
        filas.append([m, f"{f.t_fact:.2f}", f"{f.t_solve:.3f}", f"{f.nnz_LU / f.nnz:.1f}", f"{f.rss_MB:.0f}", f"{f.res_rel:.1e}", f"{f.err_rel:.1e}"])
    else:
        filas.append([m, ">300", "-", "-", "-", "-", "-"])
leyenda("Tabla 4. Solver directo con orden natural. Relleno = nnz(L+U)/nnz(A); RSS en MB.")
tabla_generica(["Matriz", "t fact. (s)", "t sol. (s)", "Relleno", "RSS (MB)", "Res. rel.", "Err. rel."], filas)
doc.add_paragraph()
doc.add_heading("Discusión", 2)
par("Con el orden natural, el costo de la factorización crece de forma muy desigual con el tamaño: va de 0.12 s en bcsstk17 a 56 s en cfd1 y 227 s en bmwcra_1, "
    "con una memoria pico de 3.3 GB y 7.9 GB respectivamente. El tiempo no depende solo de n, sino de la estructura del grafo, que determina el relleno: "
    "cant, con n similar a cfd1 (6·10⁴), se factoriza en 1.3 s frente a 56 s. Para las tres matrices de mayor tamaño (apache2, ecology2 y thermal2) "
    "no se logró factorizar en 300 s; no se determinó si se debió únicamente al tiempo o también a la presión sobre la memoria disponible. "
    "El tiempo de resolución posterior a la factorización es en cambio pequeño (menos de 1.5 s en todos los casos medidos), por lo que el costo está en la factorización.")
par("Los residuos relativos fueron de 10⁻¹⁶ a 10⁻¹³, es decir, del orden de la precisión de máquina. Sin embargo, el error relativo de cant fue 1.4·10⁻⁷ pese a un residuo de 2·10⁻¹⁴, "
    "lo que indica un mal condicionamiento de esa matriz y reproduce lo observado en el Ejercicio 1: un residuo pequeño no garantiza un error pequeño.")

doc.add_heading("Parte B – Efecto de los reordenamientos RCM y AMD", 1)
doc.add_heading("Metodología", 2)
par("Se compararon, antes de factorizar, el orden natural, Reverse Cuthill-McKee (RCM, implementación de SciPy) y minimum degree múltiple sobre el patrón de A + Aᵀ "
    "(MMD, opción MMD_AT_PLUS_A de SuperLU). Este último no es el algoritmo AMD de Amestoy, Davis y Duff, aunque pertenece a la misma familia de heurísticas de grado mínimo; "
    "por simplicidad se lo denota MMD y se lo trata como el representante de AMD en este trabajo. Se midieron además el tiempo de reordenamiento y el relleno.")
doc.add_heading("Resultados", 2)
par("La Tabla 5 reúne los resultados con los tres órdenes. Una marca «>300 s» indica que la factorización no terminó dentro del límite establecido.")
filas = []
for m in MATRICES:
    for o in ("natural", "rcm", "amd"):
        f = dd[(dd.matriz == m) & (dd.orden == o)].iloc[0]
        if f.estado == "ok":
            filas.append([m if o == "natural" else "", NOMBRE_ORD[o], f"{f.t_orden:.3f}", f"{f.t_fact:.2f}", f"{f.t_solve:.3f}",
                          f"{f.nnz_LU / f.nnz:.1f}", f"{f.rss_MB:.0f}", f"{f.res_rel:.1e}"])
        else:
            filas.append([m if o == "natural" else "", NOMBRE_ORD[o], "-", ">300", "-", "-", "-", "-"])
leyenda("Tabla 5. Efecto del reordenamiento (LU simétrico). Relleno = nnz(L+U)/nnz(A); RSS en MB.")
tabla_generica(["Matriz", "Orden", "t ord. (s)", "t fact. (s)", "t sol. (s)", "Relleno", "RSS (MB)", "Res. rel."], filas)
doc.add_paragraph()
figura("figs/directo_tiempo.png", 16, "Figura 4. Tiempo de factorización por matriz y orden (escala logarítmica). La ausencia de barra indica que no se completó en 300 s.")
figura("figs/directo_memoria.png", 16, "Figura 5. Memoria pico por matriz y orden (escala logarítmica).")


doc.add_heading("Discusión", 2)
par("El orden natural es, en general, el peor o cercano al peor. Para las tres matrices de mayor tamaño (apache2, ecology2 y thermal2) no "
    "se logró factorizar con orden natural ni con RCM en 300 s, mientras que MMD las resolvió en 5 a 90 s con residuos de 10⁻¹² a 10⁻¹⁴. "
    "Estas matrices provienen de discretizaciones sobre mallas 2D y 3D en las que el orden natural produce un perfil ancho y un "
    "relleno que crece rápidamente; esto es consistente con la teoría de factorización dispersa, donde el relleno depende de la "
    "estructura del grafo y del orden de eliminación.")
par("MMD redujo el relleno respecto del orden natural en cinco de las seis matrices en las que se pudo comparar: de 8.4·10⁶ a 0.83·10⁶ "
    "en gyro_m (≈10 veces), de 158·10⁶ a 76·10⁶ en cfd1 y de 684·10⁶ a 192·10⁶ en bmwcra_1 (≈3.6 veces). En bmwcra_1 esto llevó el "
    "tiempo de 227 s a 38 s y la memoria pico de 7.9 GB a 4.5 GB. La excepción fue cant, donde MMD aumentó el relleno de 33.7·10⁶ "
    "a 48.6·10⁶ y el tiempo de 1.3 s a 4.2 s; es posible que su numeración natural ya sea favorable, aunque no se verificó.")
par("RCM, que minimiza el ancho de banda y no el relleno, fue la opción menos consistente: mejoró a la natural en gyro_m y bmwcra_1 "
    "(esta última de 227 s a 112 s), pero empeoró en msc10848, bcsstk17 y cfd1 (de 56 s a 84 s), y fue equivalente en cant. "
    "Esto es esperable: una factorización con perfil estrecho no implica menos relleno que un orden basado en grado mínimo. "
    "Dado el patrón observado, MMD (o AMD) es el reordenamiento por defecto razonable, aunque conviene verificar el relleno con la "
    "fase simbólica cuando es posible, como ilustra cant.")

doc.add_heading("Parte C – Gradiente conjugado y precondicionadores", 1)
doc.add_heading("Metodología", 2)
par("Se empleó el gradiente conjugado de SciPy con tres variantes: sin precondicionador, con precondicionador de Jacobi "
    "(inversa de la diagonal) y con ILU(0). Esta última se programó explícitamente, siguiendo el patrón de no nulos de A, y "
    "se compiló con Numba; la aplicación del precondicionador consiste en una sustitución progresiva y otra regresiva. Como A "
    "es simétrica con patrón simétrico, el factor ILU(0) resulta simétrico, lo que es compatible con CG. El criterio de parada fue "
    "‖r_k‖/‖b‖ < 10⁻¹², con un máximo de 20 000 iteraciones, buscando un residuo del mismo orden que el del método directo. Se "
    "registraron el número de iteraciones, el tiempo de construcción del precondicionador y el tiempo de la iteración.")


doc.add_heading("Resultados", 2)
par("La Tabla 6 muestra el número de iteraciones, los tiempos y el residuo final alcanzado. «ILU(0): falla» indica que la factorización "
    "incompleta encontró un pivote no positivo y no pudo construirse.")
filas = []
for m in MATRICES:
    for p, et in (("sin precond.", "Ninguno"), ("Jacobi", "Jacobi"), ("ILU0", "ILU(0)")):
        f = cc[(cc.matriz == m) & (cc.precond == p)].iloc[0]
        if f.iter < 0:
            filas.append([m if p == "sin precond." else "", et, f"{f.t_setup:.3f}", "falla", "-", "-"])
        else:
            nota = "" if f["info"] == 0 else " (sin conv.)"
            filas.append([m if p == "sin precond." else "", et, f"{f.t_setup:.3f}", f"{int(f.iter)}{nota}", f"{f.t_cg:.2f}", f"{f.res_rel:.1e}"])
leyenda("Tabla 6. Gradiente conjugado, tolerancia 10⁻¹².")
tabla_generica(["Matriz", "Precond.", "t setup (s)", "Iteraciones", "t CG (s)", "Res. rel."], filas)
doc.add_paragraph()
figura("figs/cg_iteraciones.png", 16, "Figura 6. Iteraciones de CG según el precondicionador (escala logarítmica). Las barras ausentes de ILU(0) corresponden a fallas de la factorización.")


doc.add_heading("Discusión", 2)
par("Sin precondicionador, CG requirió entre 2 200 y más de 20 000 iteraciones. En msc10848 y bcsstk17 no convergió en el máximo "
    "de iteraciones (residuos de 1.5·10⁻⁷ y 9.4·10⁻⁹), lo que denota matrices muy mal condicionadas. El precondicionador de Jacobi fue "
    "decisivo en esas dos matrices (5 374 y 3 141 iteraciones) y redujo a menos de la mitad las iteraciones en gyro_m (8 425 a 511) "
    "y cant. En cambio, fue inútil en cfd1, ecology2 y thermal2, y casi inútil en apache2: el número de iteraciones coincide o es "
    "prácticamente el mismo. Esto es compatible con matrices de diagonal constante o casi constante, donde el escalado de Jacobi "
    "es esencialmente una multiplicación por un escalar y no modifica el espectro.")
par("ILU(0) fue el que menos iteraciones requirió donde pudo aplicarse: 145 en gyro_m, 504 en cfd1, 1 475 en bmwcra_1, 958 en apache2, "
    "2 224 en ecology2 y 2 496 en thermal2, es decir, entre 2 y 6 veces menos que con Jacobi o sin precondicionador. Sin embargo, "
    "falló en msc10848, bcsstk17 y cant por aparición de pivotes no positivos. Esto ilustra que ILU(0) no garantiza existencia para "
    "matrices SPD generales, a diferencia de lo que ocurre en M-matrices; en esos casos sería preferible una factorización con "
    "modificación diagonal o un IC con desplazamiento.")
par("Menos iteraciones no siempre implica menos tiempo: cada iteración con ILU(0) incluye dos sustituciones triangulares, "
    "inherentemente secuenciales y más costosas que un producto matriz-vector. En ecology2 el tiempo de CG con ILU(0) fue 41.5 s, "
    "mayor que los 35.3 s sin precondicionador, aunque con 3.4 veces menos iteraciones; en thermal2 fue 89.7 s frente a 76.6 s. "
    "En cambio, en cfd1, bmwcra_1 y apache2 el ILU(0) sí redujo el tiempo (por ejemplo, de 92.5 s a 27.0 s en bmwcra_1). Cabe "
    "aclarar que estos tiempos dependen de mi implementación en Numba, de un solo hilo y sin optimizar, por lo que la comparación "
    "entre precondicionadores es más confiable en iteraciones que en segundos.")
par("En las matrices apache2 y ecology2 el residuo reportado por CG alcanzó 4·10⁻¹¹ a 10⁻¹⁰ y no 10⁻¹², aunque SciPy declaró convergencia. "
    "La diferencia se debe a que el criterio de parada usa el residuo recursivo, que con el tiempo se aparta del residuo verdadero por "
    "acumulación de errores de redondeo; el residuo verdadero (el reportado en la tabla) es el que debe considerarse.")

doc.add_heading("Parte D – Comparación y criterios de elección", 1)
par("La comparación entre el método directo (con MMD) y CG, ambos al mismo nivel de residuo, no favorece siempre a un mismo enfoque.")
for t in [
    "Problemas pequeños o moderados (n ≲ 2·10⁴): el directo con reordenamiento es claramente mejor. En gyro_m resuelve en 0.05 s frente a 0.2 s del mejor CG, y es robusto frente a mal condicionamiento, donde CG sin buen precondicionador no converge (msc10848, bcsstk17).",
    "Problemas 2D de baja densidad (ecology2, thermal2): el directo con MMD es competitivo o mejor (5.4 s frente a 35 s en ecology2, y 53 s frente a 76 s en thermal2), con memoria de 2 a 4.5 GB. El relleno de estos grafos es moderado y los precondicionadores simples no mejoran el número de condición lo suficiente.",
    "Problemas 3D estructurales grandes y densos (cfd1, bmwcra_1, apache2): CG con ILU(0) resulta mucho más rápido (1.5 s frente a 18 s en cfd1, 13 s frente a 90 s en apache2, 27 s frente a 38 s en bmwcra_1). El directo exige varios GB (hasta 8.8 GB), mientras que CG necesita memoria del orden de nnz(A) más el precondicionador (no medida en este trabajo).",
    "Varios lados derechos: el directo amortiza el costo de la factorización, ya que cada resolución adicional cuesta entre 0.05 y 0.3 s en las matrices grandes, mientras que CG repite todo el costo por cada b.",
    "Matrices muy mal condicionadas donde ILU(0) falla: el directo es la opción robusta, o bien un precondicionador más sofisticado (IC con desplazamiento, multigrid algebraico).",
]:
    doc.add_paragraph(t, style="List Bullet")
par("Estas conclusiones están sujetas a las limitaciones del entorno experimental: una sola máquina de 16 GB, una única ejecución por "
    "configuración, un solver directo LU en lugar de Cholesky y un ILU(0) sin paralelizar.")


doc.add_heading("Conclusiones del Ejercicio 2", 2)
for t in [
    "El reordenamiento es esencial en métodos directos dispersos: MMD permitió resolver tres matrices que el orden natural y RCM no resolvieron en 300 s, y redujo el relleno entre 1.5 y 10 veces en cinco de seis casos comparables. RCM fue inconsistente y cant mostró que ningún reordenamiento es uniformemente mejor.",
    "El mal condicionamiento limita la precisión alcanzable con cualquier método, directo o iterativo, tal como se vio en el Ejercicio 1.",
    "CG con ILU(0) es la alternativa más rápida para los problemas grandes en 3D, pero requiere que la factorización incompleta exista y su costo por iteración es mayor.",
    "No existe un método universalmente superior: la elección depende del tamaño, de la estructura del grafo, del condicionamiento, de la memoria disponible y de la cantidad de lados derechos.",
]:
    doc.add_paragraph(t, style="List Bullet")
