from pathlib import Path

from src.graficar import graficar_todo, leer_csv

raiz = Path(__file__).resolve().parent
graficar_todo(leer_csv(raiz / "resultados" / "benchmark.csv"), raiz / "resultados", tam_img=512)