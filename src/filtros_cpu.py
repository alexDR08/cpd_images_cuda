import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _peso(i):
    # kernel binomial [1, 4, 6, 4, 1]
    if i == 0 or i == 4:
        return 1
    if i == 1 or i == 3:
        return 4
    return 6


@njit(nogil=True, cache=True)
def gris(rgb, out):
    """rgb: (H, W, 3) uint8  ->  out: (H, W) uint8"""
    h, w = out.shape
    for y in range(h):
        for x in range(w):
            r = int(rgb[y, x, 0])
            g = int(rgb[y, x, 1])
            b = int(rgb[y, x, 2])
            out[y, x] = (77 * r + 150 * g + 29 * b) >> 8


@njit(nogil=True, cache=True)
def gauss(entrada, out):
    h, w = out.shape
    for y in range(h):
        for x in range(w):
            s = 0
            for dy in range(-2, 3):
                yy = min(max(y + dy, 0), h - 1)
                wy = _peso(dy + 2)
                for dx in range(-2, 3):
                    xx = min(max(x + dx, 0), w - 1)
                    s += wy * _peso(dx + 2) * int(entrada[yy, xx])
            out[y, x] = (s + 128) >> 8      # suma de pesos = 256; +128 redondea


@njit(nogil=True, cache=True)
def sobel(entrada, out):
    h, w = out.shape
    for y in range(h):
        ym = max(y - 1, 0)
        yp = min(y + 1, h - 1)
        for x in range(w):
            xm = max(x - 1, 0)
            xp = min(x + 1, w - 1)
            a = int(entrada[ym, xm]); b = int(entrada[ym, x]); c = int(entrada[ym, xp])
            d = int(entrada[y, xm]);                           f = int(entrada[y, xp])
            g = int(entrada[yp, xm]); hh = int(entrada[yp, x]); i = int(entrada[yp, xp])
            gx = (c + 2 * f + i) - (a + 2 * d + g)
            gy = (g + 2 * hh + i) - (a + 2 * b + c)
            m = int(math.sqrt(float(gx * gx + gy * gy)))
            out[y, x] = 255 if m > 255 else m


@njit(nogil=True, cache=True)
def procesar_lote(rgb, out, inicio, fin):
    """Procesa las imágenes [inicio, fin) del lote.
    rgb: (N, H, W, 3) uint8 | out: (N, H, W) uint8"""
    h = out.shape[1]
    w = out.shape[2]
    g = np.empty((h, w), dtype=np.uint8)
    b = np.empty((h, w), dtype=np.uint8)
    for k in range(inicio, fin):
        gris(rgb[k], g)
        gauss(g, b)
        sobel(b, out[k])


def procesar_secuencial(rgb, out):
    """Versión 1: un solo hilo procesa todo el lote."""
    procesar_lote(rgb, out, 0, rgb.shape[0])