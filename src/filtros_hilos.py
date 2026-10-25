from concurrent.futures import ThreadPoolExecutor
from .filtros_cpu import procesar_lote

def repartir(n, hilos):
    """Reparte n imágenes en 'hilos' rangos [inicio, fin) lo más parejos posible."""
    base, extra = divmod(n, hilos)       # si no es divisible, sobran 'extra'
    rangos, inicio = [], 0
    for h in range(hilos):
        fin = inicio + base + (1 if h < extra else 0)   # los primeros reciben una de más
        rangos.append((inicio, fin))
        inicio = fin
    return rangos


def procesar_con_hilos(rgb, out, hilos):
    """Versión 2: reparte el lote entre hilos y espera a que todos terminen."""
    with ThreadPoolExecutor(max_workers=hilos) as pool:
        tareas = [pool.submit(procesar_lote, rgb, out, i0, i1)
                  for i0, i1 in repartir(rgb.shape[0], hilos)]
        for t in tareas:
            t.result()