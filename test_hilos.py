import statistics
import time
import numpy as np
from src.cargar_datos import cargar_lote
from src.filtros_cpu import procesar_secuencial
from src.filtros_hilos import procesar_con_hilos

REPS = 5


def medir(funcion):
    tiempos = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - t0)
    return statistics.median(tiempos)


lote = cargar_lote(n=500)
out_ref = np.zeros(lote.shape[:3], dtype=np.uint8)
out = np.zeros_like(out_ref)

procesar_secuencial(lote[:2], out[:2])          # calentamiento (compilación JIT)

t_sec = medir(lambda: procesar_secuencial(lote, out_ref))
print(f"{'Config':<12}{'Tiempo (s)':>12}{'Aceleración':>13}{'Eficiencia':>12}{'Igual?':>8}")
print(f"{'Secuencial':<12}{t_sec:>12.4f}{1.0:>13.2f}{'-':>12}{'-':>8}")

for h in [2, 4, 8]:
    t = medir(lambda: procesar_con_hilos(lote, out, h))
    igual = np.array_equal(out, out_ref)
    print(f"{h:<2} hilos    {t:>12.4f}{t_sec / t:>13.2f}{t_sec / t / h:>12.0%}{str(igual):>8}")