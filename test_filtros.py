import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.cargar_datos import cargar_lote
from src.filtros_cpu import gris, gauss, sobel, procesar_secuencial

lote = cargar_lote(n=100)

# --- 1) Revisión visual con la primera imagen ---
rgb = lote[0]
g = np.zeros(rgb.shape[:2], dtype=np.uint8)
b = np.zeros_like(g)
s = np.zeros_like(g)
gris(rgb, g)
gauss(g, b)
sobel(b, s)

tira = np.hstack([g, b, s])                      # gris | gauss | sobel
Path("resultados").mkdir(exist_ok=True)
Image.fromarray(tira).save("resultados/ejemplo_filtros.png")
print("Imagen guardada en resultados/ejemplo_filtros.png")

# --- 2) Tiempo del lote completo (la primera llamada compila, no se cuenta) ---
out = np.zeros(lote.shape[:3], dtype=np.uint8)
procesar_secuencial(lote[:2], out[:2])           # calentamiento (compilación JIT)
t0 = time.perf_counter()
procesar_secuencial(lote, out)
print(f"Secuencial: {len(lote)} imágenes en {time.perf_counter() - t0:.3f} s")