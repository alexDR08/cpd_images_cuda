import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # solo guarda PNG, no abre ventanas
import matplotlib.pyplot as plt

# gris = secuencial; azules (más oscuro = más hilos) = multihilo;
# naranja = GPU pura; verde azulado = GPU real (con PCIe)
TINTA, TINTA_SEC = "#0b0b0b", "#52514e"
GRID, EJE, GRIS = "#e1e0d9", "#c3c2b7", "#898781"
AZULES = ["#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
NARANJA, AQUA, AZUL = "#eb6834", "#1baf7a", "#2a78d6"


def leer_csv(ruta):
    with open(ruta, newline="", encoding="utf-8") as fh:
        return [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(fh)]


def _hay(v):
    return not math.isnan(v)


def _hilos_en(filas):
    return sorted(int(k.split("_")[-1]) for k in filas[0] if k.startswith("t_hilos_"))


def _estilo(ax, xlabel, ylabel, titulo):
    ax.set_xlabel(xlabel, color=TINTA_SEC)
    ax.set_ylabel(ylabel, color=TINTA_SEC)
    ax.set_title(titulo, color=TINTA, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)
    ax.tick_params(colors=TINTA_SEC)


def _linea(ax, xs, ys, color, label, marcador="o"):
    """Dibuja una serie y escribe su nombre al final de la línea."""
    pares = [(x, y) for x, y in zip(xs, ys) if _hay(y)]
    if not pares:
        return
    px, py = zip(*pares)
    ax.plot(px, py, "-", marker=marcador, color=color, linewidth=2, markersize=7,
            markeredgecolor="white", markeredgewidth=1.5, label=label)
    ax.annotate(label, (px[-1], py[-1]), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=9, color=TINTA_SEC)


def _ejes_log(ax, xs):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlim(min(xs) * 0.85, max(xs) * 2.6)      # espacio a la derecha para etiquetas


def graficar_todo(filas, carpeta, tam_img=None):
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)
    hilos = _hilos_en(filas)
    xs = [int(f["tamano"]) for f in filas]
    sub = f" ({tam_img}x{tam_img} px)" if tam_img else ""

    # ---- 1) Tamaño del problema vs aceleración ----
    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor="white")
    for i, k in enumerate(hilos):
        _linea(ax, xs, [f[f"acel_hilos_{k}"] for f in filas],
               AZULES[min(i, len(AZULES) - 1)], f"CPU {k} hilos", "s")
    _linea(ax, xs, [f["acel_pura"] for f in filas], NARANJA, "GPU pura", "o")
    _linea(ax, xs, [f["acel_real"] for f in filas], AQUA, "GPU real (con PCIe)", "D")
    ax.axhline(1, color=GRIS, linewidth=1, linestyle="--")
    ax.annotate("Secuencial (1x)", (xs[-1], 1), xytext=(9, 7), textcoords="offset points",
                va="center", fontsize=9, color=TINTA_SEC)
    _ejes_log(ax, xs)
    _estilo(ax, "Tamaño del problema (número de imágenes)",
            "Aceleración respecto al secuencial (veces, escala log)",
            f"Aceleración según el tamaño del problema{sub}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(carpeta / "grafica_aceleracion.png", dpi=150)
    plt.close(fig)

    # ---- 2) Tamaño del problema vs tiempo ----
    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor="white")
    _linea(ax, xs, [f["t_secuencial"] for f in filas], GRIS, "CPU secuencial")
    for i, k in enumerate(hilos):
        _linea(ax, xs, [f[f"t_hilos_{k}"] for f in filas],
               AZULES[min(i, len(AZULES) - 1)], f"CPU {k} hilos", "s")
    _linea(ax, xs, [f["t_total_gpu"] for f in filas], AQUA, "GPU total (con PCIe)", "D")
    _linea(ax, xs, [f["t_gpu"] for f in filas], NARANJA, "GPU solo kernels", "o")
    _ejes_log(ax, xs)
    _estilo(ax, "Tamaño del problema (número de imágenes)", "Tiempo (s, escala log)",
            f"Tiempo de ejecución según el tamaño del problema{sub}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(carpeta / "grafica_tiempos.png", dpi=150)
    plt.close(fig)

    # ---- 3) Desglose del tiempo de la GPU: PCIe vs cómputo ----
    con_gpu = [f for f in filas if _hay(f["t_gpu"])]
    if con_gpu:
        fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor="white")
        et = [str(int(f["tamano"])) for f in con_gpu]
        h2d = [f["t_h2d"] * 1000 for f in con_gpu]
        ker = [f["t_gpu"] * 1000 for f in con_gpu]
        d2h = [f["t_d2h"] * 1000 for f in con_gpu]
        borde = dict(edgecolor="white", linewidth=2, width=0.6)    # 2 px entre segmentos
        ax.bar(et, h2d, color=AZUL, label="Transferencia H2D", **borde)
        ax.bar(et, ker, bottom=h2d, color=NARANJA, label="Kernels (GPU)", **borde)
        ax.bar(et, d2h, bottom=[a + b for a, b in zip(h2d, ker)], color=AQUA,
               label="Transferencia D2H", **borde)
        for i, (a, b, c) in enumerate(zip(h2d, ker, d2h)):
            ax.annotate(f"{a + b + c:.0f} ms", (i, a + b + c), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=9, color=TINTA_SEC)
        _estilo(ax, "Tamaño del problema (número de imágenes)", "Tiempo (ms)",
                f"¿En qué se va el tiempo de la GPU?{sub}")
        ax.set_ylim(0, max(a + b + c for a, b, c in zip(h2d, ker, d2h)) * 1.12)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(carpeta / "grafica_desglose_gpu.png", dpi=150)
        plt.close(fig)

    _tablas(filas, hilos, carpeta)
    print(f"Gráficas y tablas guardadas en: {carpeta}")


def _fmt(v, dec=4, sufijo=""):
    return f"{v:.{dec}f}{sufijo}" if _hay(v) else "n/d"


def _tablas(filas, hilos, carpeta):
    # Tabla comparativa de tiempos (segundos)
    enc = (["Imágenes", "Secuencial"] + [f"{k} hilos" for k in hilos] +
           ["Transferencia H2D", "GPU (kernels)", "Transferencia D2H", "GPU total con PCIe"])
    out = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for f in filas:
        celdas = [str(int(f["tamano"])), _fmt(f["t_secuencial"])]
        celdas += [_fmt(f[f"t_hilos_{k}"]) for k in hilos]
        celdas += [_fmt(f["t_h2d"]), _fmt(f["t_gpu"]), _fmt(f["t_d2h"]), _fmt(f["t_total_gpu"])]
        out.append("| " + " | ".join(celdas) + " |")
    (carpeta / "tabla_tiempos.md").write_text(
        "Tiempos en segundos (mediana de las repeticiones)\n\n" + "\n".join(out) + "\n",
        encoding="utf-8")

    # Tabla de métricas de aceleración
    enc = (["Imágenes"] + [f"{k} hilos" for k in hilos] +
           ["GPU pura", "GPU real (con PCIe)", "PCIe como % del total GPU"])
    out = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for f in filas:
        pcie = (f["t_h2d"] + f["t_d2h"]) / f["t_total_gpu"] * 100 if _hay(f["t_gpu"]) else float("nan")
        celdas = [str(int(f["tamano"]))]
        celdas += [_fmt(f[f"acel_hilos_{k}"], 2, "x") for k in hilos]
        celdas += [_fmt(f["acel_pura"], 1, "x"), _fmt(f["acel_real"], 1, "x"), _fmt(pcie, 1, " %")]
        out.append("| " + " | ".join(celdas) + " |")
    (carpeta / "tabla_aceleracion.md").write_text(
        "Aceleración = T_secuencial / T_versión\n\n" + "\n".join(out) + "\n", encoding="utf-8")