import csv
import statistics
import time

import numpy as np
from numba import cuda

from .filtros_cpu import procesar_secuencial
from .filtros_cuda import medir_gpu, procesar_gpu
from .filtros_hilos import procesar_con_hilos

NAN = float("nan")


def medir(funcion, repeticiones):
    """Mediana de varias ejecuciones (resiste picos de ruido del sistema)."""
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - t0)
    return statistics.median(tiempos)


def calentar(rgb_chico):
    """Compila (JIT) las tres versiones para que no se cuente en las mediciones."""
    n, h, w, _ = rgb_chico.shape
    out = np.zeros((n, h, w), dtype=np.uint8)
    procesar_secuencial(rgb_chico, out)
    procesar_con_hilos(rgb_chico, out, 2)
    procesar_gpu(rgb_chico)


def correr_benchmark(rgb_total, tamanos, hilos, repeticiones):
    _, h, w, _ = rgb_total.shape
    filas = []
    for n in tamanos:
        rgb = np.ascontiguousarray(rgb_total[:n])
        out = np.zeros((n, h, w), dtype=np.uint8)
        print(f"\n=== {n} imágenes de {h}x{w} ===")

        fila = {"tamano": n}
        fila["t_secuencial"] = medir(lambda: procesar_secuencial(rgb, out), repeticiones)
        for k in hilos:
            fila[f"t_hilos_{k}"] = medir(lambda: procesar_con_hilos(rgb, out, k), repeticiones)

        # ¿cabe en la VRAM? (RGB + gris + gauss + salida = 6 bytes por píxel)
        necesaria = n * h * w * 6
        libre, _ = cuda.current_context().get_memory_info()
        gpu_ok = necesaria < 0.9 * libre
        if gpu_ok:
            t, _ = medir_gpu(rgb, repeticiones)
            fila.update(t_h2d=t["h2d"], t_gpu=t["kernel"], t_d2h=t["d2h"], t_total_gpu=t["total"])
            fila["acel_pura"] = fila["t_secuencial"] / t["kernel"]
            fila["acel_real"] = fila["t_secuencial"] / t["total"]
        else:
            print(f"  [aviso] no cabe en la GPU ({necesaria / 2**20:.0f} MiB necesarios, "
                  f"{libre / 2**20:.0f} MiB libres): se omite la GPU")
            fila.update(t_h2d=NAN, t_gpu=NAN, t_d2h=NAN, t_total_gpu=NAN,
                        acel_pura=NAN, acel_real=NAN)

        for k in hilos:
            fila[f"acel_hilos_{k}"] = fila["t_secuencial"] / fila[f"t_hilos_{k}"]
        filas.append(fila)
        _imprimir(fila, hilos)
    return filas


def _imprimir(f, hilos):
    print(f"  Secuencial        : {f['t_secuencial']:.4f} s")
    for k in hilos:
        print(f"  {k} hilos           : {f[f't_hilos_{k}']:.4f} s  ({f[f'acel_hilos_{k}']:.2f}x)")
    if f["t_gpu"] == f["t_gpu"]:                     # no es NaN
        print(f"  Transferencia H2D : {f['t_h2d']:.4f} s")
        print(f"  GPU (kernels)     : {f['t_gpu']:.4f} s  (pura {f['acel_pura']:.1f}x)")
        print(f"  Transferencia D2H : {f['t_d2h']:.4f} s")
        print(f"  GPU total + PCIe  : {f['t_total_gpu']:.4f} s  (real {f['acel_real']:.1f}x)")


def guardar_csv(filas, ruta):
    ruta.parent.mkdir(exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\nResultados guardados en: {ruta}")