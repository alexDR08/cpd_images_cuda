import statistics
import time

import numpy as np
from numba import cuda

from src.cargar_datos import cargar_lote
from src.filtros_cpu import procesar_secuencial
from src.filtros_cuda import medir_gpu, procesar_gpu
from src.filtros_hilos import procesar_con_hilos

print("GPU:", cuda.get_current_device().name)

lote = cargar_lote(n=500)
out_ref = np.zeros(lote.shape[:3], dtype=np.uint8)
procesar_secuencial(lote, out_ref)

# 1) Validación: la GPU debe dar el mismo resultado que la CPU
out_gpu = procesar_gpu(lote)
dif = np.abs(out_gpu.astype(np.int16) - out_ref.astype(np.int16))
print(f"Diferencia máxima GPU vs CPU: {dif.max()} | píxeles distintos: {(dif > 0).sum()}")

# 2) Tiempos
def medir(f, reps=5):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return statistics.median(t)

out = np.zeros_like(out_ref)
t_sec = medir(lambda: procesar_secuencial(lote, out))
t_8h = medir(lambda: procesar_con_hilos(lote, out, 8))
t, _ = medir_gpu(lote)

print(f"\nSecuencial        : {t_sec:.4f} s")
print(f"8 hilos           : {t_8h:.4f} s  ({t_sec / t_8h:.2f}x)")
print(f"Transferencia H2D : {t['h2d']:.4f} s")
print(f"GPU (kernels)     : {t['kernel']:.4f} s  (aceleración pura {t_sec / t['kernel']:.1f}x)")
print(f"Transferencia D2H : {t['d2h']:.4f} s")
print(f"GPU total + PCIe  : {t['total']:.4f} s  (aceleración real {t_sec / t['total']:.1f}x)")