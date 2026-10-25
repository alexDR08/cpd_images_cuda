import time
from src.cargar_datos import cargar_lote

t0 = time.perf_counter()
lote = cargar_lote(n=100)
print("Forma:", lote.shape, "| tipo:", lote.dtype)
print(f"Memoria: {lote.nbytes / 2**20:.0f} MiB | carga: {time.perf_counter() - t0:.1f} s")