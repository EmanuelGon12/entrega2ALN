import pandas as pd
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

d = pd.read_csv("ej1_resultados.csv")
d["kappa"] = d["kappa"].astype(float)
metodos = ["LU sin pivoteo", "LU pivoteo parcial", "inv(A)", "LAPACK lu", "QR"]

doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.5)
    s.top_margin = s.bottom_margin = Cm(2.5)
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
for h in ("Heading 1", "Heading 2"):
    doc.styles[h].font.name = "Times New Roman"


def par(texto, justificar=True):
    p = doc.add_paragraph(texto)
    if justificar:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def figura(ruta, ancho, pie):
    doc.add_picture(ruta, width=Cm(ancho))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(pie)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True
    p.runs[0].font.size = Pt(10)


def tabla(tipo, n):
    cab = ["κ", "Método", "t fact. (s)", "Error rel.", "Residuo rel.", "Error rel. (b pert.)"]
    t = doc.add_table(rows=1, cols=len(cab))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(cab):
        t.rows[0].cells[i].text = c
    for k in (1e2, 1e6, 1e10):
        for m in metodos:
            r = d[(d.tipo == tipo) & (d.n == n) & (d.kappa == k) & (d.metodo == m)].iloc[0]
            fila = t.add_row().cells
            vals = [f"{k:.0e}", m, f"{r.t_fact:.3f}", f"{r.err_rel:.2e}", f"{r.res_rel:.2e}", f"{r.err_rel_pert:.2e}"]
            for i, v in enumerate(vals):
                fila[i].text = v
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)


tit = doc.add_heading("Tarea 2 – Álgebra Lineal Numérica 2026", 0)
tit.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("Informe")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.runs[0].bold = True

doc.add_heading("1. Introducción", 1)
par("En este informe se estudia la resolución de sistemas lineales densos Ax = b mediante distintas estrategias de "
    "factorización y se analiza su eficiencia y estabilidad numérica en función del tamaño del sistema y del número de "
    "condición de la matriz (Ejercicio 1), y la resolución de sistemas dispersos simétricos definidos positivos de la colección "
    "SuiteSparse mediante un método directo con reordenamientos RCM y AMD y mediante gradiente conjugado precondicionado (Ejercicio 2).")

doc.add_heading("Ejercicio 1", 1)
doc.add_heading("2. Metodología", 1)
doc.add_heading("2.1 Generación de las matrices", 2)
par("Dado que para una matriz S no singular las matrices A y SAS⁻¹ comparten los mismos valores propios, y que una matriz "
    "ortogonal Q verifica Q⁻¹ = Qᵀ, se construyeron matrices A = Q Λ Qᵀ con Q ortogonal obtenida de la factorización QR de una "
    "matriz gaussiana aleatoria y Λ = diag(λ₁,…,λₙ) con valores propios espaciados logarítmicamente entre 1 y κ. Como Q es "
    "ortogonal, el número de condición espectral de A es exactamente κ. Se consideraron tres tamaños, n ∈ {512, 1024, 2048}, y "
    "tres números de condición, κ ∈ {10², 10⁶, 10¹⁰}.")
par("Se trabajó con dos familias: (i) matrices simétricas definidas positivas (SPD), con todos los λᵢ positivos; y (ii) matrices "
    "simétricas indefinidas, en las que cada λᵢ recibe un signo aleatorio. La segunda familia se incluyó porque la factorización "
    "LU sin pivoteo es estable para matrices SPD, y por lo tanto no permite distinguir el efecto del pivoteo.")
doc.add_heading("2.2 Métodos comparados", 2)
par("Se implementaron en Python (NumPy/SciPy) la factorización LU sin pivoteo y la factorización LU con pivoteo parcial, ambas en "
    "su variante por productos externos (right-looking), seguidas de sustitución progresiva y regresiva. Se compararon con: el "
    "cálculo explícito de A⁻¹ (numpy.linalg.inv) y posterior producto A⁻¹b; la rutina lu_factor/lu_solve de LAPACK; y la "
    "factorización QR de LAPACK, resolviendo Rx = Qᵀb.")
doc.add_heading("2.3 Experimentos y métricas", 2)
par("Se fijó la solución exacta x* = (1,…,1)ᵀ y se definió b = Ax*. Para cada método se midieron el tiempo de factorización, el "
    "error relativo ‖x − x*‖₂/‖x*‖₂ y el residuo relativo ‖b − Ax‖₂/(‖A‖₂‖x‖₂ + ‖b‖₂). Posteriormente se repitió el experimento "
    "con un vector perturbado b̃ = b + δ, con δ de dirección aleatoria y norma relativa ‖δ‖/‖b‖ = 10⁻⁸. Los resultados completos "
    "se encuentran en el archivo ej1_resultados.csv y el código en ej1.py. Cada configuración se ejecutó una única vez con "
    "semilla fija, por lo que los tiempos tienen la variabilidad propia de una medición única.")

doc.add_heading("3. Resultados", 1)
par("Las Tablas 1 y 2 reúnen los resultados para n = 2048. Las Figuras 1 y 2 muestran el error y el residuo relativos en "
    "función de κ, y la Figura 3 el tiempo de factorización en función de n.")
doc.add_paragraph("Tabla 1. Matrices SPD, n = 2048.").runs[0].italic = True
tabla("spd", 2048)
doc.add_paragraph()
doc.add_paragraph("Tabla 2. Matrices simétricas indefinidas, n = 2048.").runs[0].italic = True
tabla("indef", 2048)
doc.add_paragraph()
figura("figs/error_vs_kappa.png", 15.5, "Figura 1. Error relativo en x en función de κ(A), n = 2048.")
figura("figs/residuo_vs_kappa.png", 15.5, "Figura 2. Residuo relativo en función de κ(A), n = 2048.")
figura("figs/tiempo_vs_n.png", 9.5, "Figura 3. Tiempo de factorización en función de n (matrices SPD, κ = 10⁶).")

doc.add_heading("4. Discusión", 1)
doc.add_heading("4.1 Estabilidad numérica", 2)
par("El error relativo crece de forma aproximadamente proporcional a κ(A) para todos los métodos: en matrices SPD de n = 2048 pasa de "
    "~10⁻¹⁴ con κ = 10² a ~10⁻⁷ con κ = 10¹⁰. Esto es coherente con la cota error ≲ κ(A)·u, con u ≈ 1.1·10⁻¹⁶ la unidad de "
    "redondeo, y refleja que la pérdida de precisión es una propiedad del condicionamiento del problema y no del algoritmo.")
par("La diferencia entre métodos se aprecia en el residuo. Las factorizaciones LU con pivoteo, LAPACK y QR mantienen un residuo "
    "relativo del orden de la precisión de máquina (10⁻¹⁶) aun con κ = 10¹⁰, es decir que son retroestables: la solución "
    "calculada es la exacta de un sistema muy cercano al original. La inversión explícita, en cambio, deteriora el residuo hasta "
    "~10⁻⁸ para κ = 10¹⁰ en matrices SPD (≈2·10⁻⁸ en n = 2048) y ~10⁻⁷ en las indefinidas, y presenta errores de 5 a 10 veces mayores "
    "que los demás métodos. Calcular A⁻¹ y multiplicar por b no es retroestable y debe evitarse para resolver sistemas.")
par("En el caso SPD, LU sin pivoteo y LU con pivoteo parcial arrojan resultados idénticos en todas las métricas, lo que es "
    "consistente con que para matrices SPD la eliminación gaussiana es estable sin necesidad de intercambios de filas. En las matrices "
    "indefinidas la situación cambia: sin pivoteo el error es entre 1 y 3 órdenes de magnitud mayor que con pivoteo (por ejemplo, "
    "2.7·10⁻⁴ frente a 8.9·10⁻⁷ para n = 2048 y κ = 10¹⁰) y el residuo es entre 10² y 10⁴ veces mayor (p. ej. 1.1·10⁻¹² frente a "
    "6.3·10⁻¹⁵ para κ = 10² y n = 2048). Esto se debe a la aparición de pivotes pequeños que generan multiplicadores grandes y crecimiento de "
    "los elementos durante la eliminación. Cabe notar que en estos experimentos el efecto es moderado y no catastrófico, "
    "dado que el pivote nulo exacto es improbable en matrices aleatorias; existen matrices (por ejemplo con a₁₁ = 0) para las que "
    "LU sin pivoteo falla directamente.")
doc.add_heading("4.2 Perturbaciones en b", 2)
par("Al perturbar b con una magnitud relativa de 10⁻⁸, el error relativo en x resultó prácticamente idéntico para todos los métodos "
    "(Tablas 1 y 2, última columna): ≈10⁻⁷ para κ = 10², ≈10⁻⁴ para κ = 10⁶ y de orden 1 (≈2) para κ = 10¹⁰. El factor de "
    "amplificación observado (‖δx‖/‖x‖ dividido por ‖δb‖/‖b‖) fue del orden de 10¹, 10⁴ y 10⁸ respectivamente, siempre por debajo de la cota "
    "teórica κ(A). Para κ = 10¹⁰ y perturbación 10⁻⁸ la cota supera la unidad y la solución carece de dígitos significativos "
    "correctos. Los residuos, en cambio, se mantienen en el orden de la precisión de máquina para los métodos retroestables, "
    "lo que ilustra que un residuo pequeño no implica un error pequeño en sistemas mal condicionados. Ningún algoritmo puede "
    "mitigar este efecto: es una limitación intrínseca del problema.")
doc.add_heading("4.3 Eficiencia", 2)
par("Las cuatro estrategias basadas en factorización tienen costo O(n³) y los tiempos crecen en consecuencia al duplicar n "
    "(Figura 3). La rutina de LAPACK es la más rápida por amplio margen (0.06 s para n = 2048), seguida por la inversión (0.14 s), "
    "QR (≈0.4 s) y las implementaciones propias de LU (≈3.4 s). La brecha de unas 55 veces entre las LU propias y LAPACK no se debe a "
    "diferencias algorítmicas, ya que el conteo de operaciones es el mismo (≈2n³/3), sino a que LAPACK emplea algoritmos por bloques "
    "que aprovechan la jerarquía de memoria caché y BLAS nivel 3 optimizado, mientras que la implementación propia realiza "
    "n actualizaciones de rango uno, limitadas por el ancho de banda de memoria y con el bucle externo interpretado. "
    "La factorización QR cuesta aproximadamente el doble de operaciones que LU (≈4n³/3), con una ventaja en robustez que, "
    "para estas matrices, no se tradujo en una mejora apreciable de la precisión respecto de LU con pivoteo.")
par("El tiempo de resolución posterior a la factorización (sustituciones o producto matriz-vector, O(n²)) (1 a 3 ms para n = 2048) es de uno a tres órdenes de "
    "magnitud menor que el de la factorización, lo que hace conveniente reutilizar los factores cuando hay múltiples lados "
    "derechos. Obsérvese además que la inversión no ofrece ventajas en precisión, y que su costo aparente en la tabla "
    "(menor que el de QR) proviene de que se apoya en LAPACK optimizado, no de una menor complejidad: calcular A⁻¹ requiere "
    "≈2n³ operaciones, tres veces más que una factorización LU.")

doc.add_heading("5. Conclusiones del Ejercicio 1", 1)
for t in [
    "El error en la solución está gobernado por κ(A) y es común a todos los métodos; la elección del algoritmo afecta principalmente al residuo y a la robustez.",
    "LU con pivoteo parcial, LU de LAPACK y QR son retroestables; la inversión explícita de A no lo es y debe evitarse.",
    "El pivoteo parcial es innecesario para matrices SPD, pero resulta importante en matrices indefinidas.",
    "Una implementación eficiente depende más del uso de bibliotecas optimizadas (bloques, BLAS 3) que del algoritmo en sí.",
]:
    doc.add_paragraph(t, style="List Bullet")

exec(open("informe_ej2.py").read())

doc.save("informe_tarea2.docx")
