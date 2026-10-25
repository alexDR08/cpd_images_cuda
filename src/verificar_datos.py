from pathlib import Path
from collections import Counter
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent / "data" / "dataset_xray" / "chest_xray"
print("Buscando en:", RAIZ, "| existe:", RAIZ.exists())
EXT = {".jpeg", ".jpg", ".png"}

# Ignora archivos basura de macOS por si quedara alguno
archivos = sorted(p for p in RAIZ.rglob("*")
                  if p.suffix.lower() in EXT
                  and not p.name.startswith("._")
                  and "__MACOSX" not in p.parts)
print(f"Total de imágenes: {len(archivos)}")

conteo = Counter(f"{p.parent.parent.name}/{p.parent.name}" for p in archivos)
for carpeta, n in sorted(conteo.items()):
    print(f"  {carpeta:<18} {n}")

muestra = archivos[::max(1, len(archivos) // 200)]
tamanos, modos = Counter(), Counter()
for p in muestra:
    with Image.open(p) as im:
        modos[im.mode] += 1
        tamanos[im.size] += 1
print("\nModos de color:", dict(modos))
print("Tamaños distintos en la muestra:", len(tamanos))
print("Ejemplos:", list(tamanos)[:5])