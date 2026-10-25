from pathlib import Path

from src.benchmark import calentar, correr_benchmark, guardar_csv
from src.cargar_datos import cargar_lote

TAMANOS = [100, 250, 500, 1000, 1500]    # tamaños de problema (número de imágenes)
HILOS = [2, 4, 8]
REPS = 5

if __name__ == "__main__":
    lote = cargar_lote(n=max(TAMANOS))
    print("Calentando (compilación)...")
    calentar(lote[:2])
    filas = correr_benchmark(lote, TAMANOS, HILOS, REPS)
    guardar_csv(filas, Path(__file__).resolve().parent / "resultados" / "benchmark.csv")