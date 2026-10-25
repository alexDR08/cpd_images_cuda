from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# Carpeta que contiene train/, test/ y val/ (anclada a la raíz del proyecto)
RAIZ_DATOS = Path(__file__).resolve().parent.parent / "data" / "dataset_xray" / "chest_xray"
EXTENSIONES = {".jpeg", ".jpg", ".png"}


def listar_imagenes(carpeta=RAIZ_DATOS):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")
    archivos = sorted(p for p in carpeta.rglob("*")
                      if p.suffix.lower() in EXTENSIONES
                      and not p.name.startswith("._")
                      and "__MACOSX" not in p.parts)
    if not archivos:
        raise FileNotFoundError(f"No encontré imágenes en {carpeta}")
    return archivos


def _leer(ruta, tam):
    # convert("RGB"): casi todas vienen en 1 canal (gris); así el paso
    # "RGB -> gris" del pipeline siempre recibe 3 canales.
    with Image.open(ruta) as im:
        im = im.convert("RGB").resize((tam, tam), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def cargar_lote(n, tam=512, semilla=42, carpeta=RAIZ_DATOS):
    """Devuelve un arreglo (N, tam, tam, 3) uint8 contiguo en memoria."""
    archivos = listar_imagenes(carpeta)
    orden = np.random.default_rng(semilla).permutation(len(archivos))[:n]
    elegidos = [archivos[i] for i in orden]
    print(f"Leyendo {len(elegidos)} de {len(archivos)} imágenes ({tam}x{tam})...")
    with ThreadPoolExecutor(max_workers=8) as pool:   # PIL libera el GIL al decodificar
        imgs = list(pool.map(lambda p: _leer(p, tam), elegidos))
    return np.ascontiguousarray(np.stack(imgs))