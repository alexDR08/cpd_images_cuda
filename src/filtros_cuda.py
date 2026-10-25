import math
import numpy as np
import statistics
import time
from numba import cuda

BLOQUE = (16, 16, 1)

@cuda.jit(device=True)
def _peso(i):
    if i == 0 or i == 4:
        return 1
    if i == 1 or i == 3:
        return 4
    return 6


@cuda.jit
def k_gris(rgb, out):
    x, y, n = cuda.grid(3)
    if n < out.shape[0] and y < out.shape[1] and x < out.shape[2]:
        r = int(rgb[n, y, x, 0])
        g = int(rgb[n, y, x, 1])
        b = int(rgb[n, y, x, 2])
        out[n, y, x] = (77 * r + 150 * g + 29 * b) >> 8


@cuda.jit
def k_gauss(entrada, out):
    x, y, n = cuda.grid(3)
    h = out.shape[1]
    w = out.shape[2]
    if n < out.shape[0] and y < h and x < w:
        s = 0
        for dy in range(-2, 3):
            yy = min(max(y + dy, 0), h - 1)
            wy = _peso(dy + 2)
            for dx in range(-2, 3):
                xx = min(max(x + dx, 0), w - 1)
                s += wy * _peso(dx + 2) * int(entrada[n, yy, xx])
        out[n, y, x] = (s + 128) >> 8


@cuda.jit
def k_sobel(entrada, out):
    x, y, n = cuda.grid(3)
    h = out.shape[1]
    w = out.shape[2]
    if n < out.shape[0] and y < h and x < w:
        ym = max(y - 1, 0)
        yp = min(y + 1, h - 1)
        xm = max(x - 1, 0)
        xp = min(x + 1, w - 1)
        a = int(entrada[n, ym, xm]); b = int(entrada[n, ym, x]); c = int(entrada[n, ym, xp])
        d = int(entrada[n, y, xm]);                              f = int(entrada[n, y, xp])
        g = int(entrada[n, yp, xm]); hh = int(entrada[n, yp, x]); i = int(entrada[n, yp, xp])
        gx = (c + 2 * f + i) - (a + 2 * d + g)
        gy = (g + 2 * hh + i) - (a + 2 * b + c)
        m = int(math.sqrt(float(gx * gx + gy * gy)))
        out[n, y, x] = 255 if m > 255 else m


def _lanzar(d_rgb, d_gris, d_blur, d_out):
    """Lanza los 3 kernels sobre todo el lote (sin sincronizar)."""
    n, h, w = d_out.shape
    grid = ((w + BLOQUE[0] - 1) // BLOQUE[0],
            (h + BLOQUE[1] - 1) // BLOQUE[1],
            n)
    k_gris[grid, BLOQUE](d_rgb, d_gris)
    k_gauss[grid, BLOQUE](d_gris, d_blur)
    k_sobel[grid, BLOQUE](d_blur, d_out)


def procesar_gpu(rgb):
    """Procesa un lote completo en la GPU y devuelve el resultado en la RAM."""
    n, h, w, _ = rgb.shape
    d_rgb = cuda.to_device(rgb)
    d_gris = cuda.device_array((n, h, w), dtype=np.uint8)
    d_blur = cuda.device_array((n, h, w), dtype=np.uint8)
    d_out = cuda.device_array((n, h, w), dtype=np.uint8)
    _lanzar(d_rgb, d_gris, d_blur, d_out)
    cuda.synchronize()
    return d_out.copy_to_host()


def medir_gpu(rgb, repeticiones=5):
    """Mide H2D, kernels y D2H por separado (mediana). Devuelve (tiempos, resultado)."""
    n, h, w, _ = rgb.shape
    d_rgb = cuda.device_array(rgb.shape, dtype=np.uint8)
    d_gris = cuda.device_array((n, h, w), dtype=np.uint8)
    d_blur = cuda.device_array((n, h, w), dtype=np.uint8)
    d_out = cuda.device_array((n, h, w), dtype=np.uint8)
    h_out = np.empty((n, h, w), dtype=np.uint8)

    # calentamiento (compilación JIT + arranque del contexto CUDA): no se cuenta
    d_rgb.copy_to_device(rgb)
    _lanzar(d_rgb, d_gris, d_blur, d_out)
    d_out.copy_to_host(h_out)
    cuda.synchronize()

    t_h2d, t_ker, t_d2h = [], [], []
    for _ in range(repeticiones):
        cuda.synchronize()
        t0 = time.perf_counter()
        d_rgb.copy_to_device(rgb)                 # RAM -> VRAM (PCIe)
        cuda.synchronize()
        t1 = time.perf_counter()
        _lanzar(d_rgb, d_gris, d_blur, d_out)     # cómputo en la GPU
        cuda.synchronize()
        t2 = time.perf_counter()
        d_out.copy_to_host(h_out)                 # VRAM -> RAM (PCIe)
        cuda.synchronize()
        t3 = time.perf_counter()
        t_h2d.append(t1 - t0)
        t_ker.append(t2 - t1)
        t_d2h.append(t3 - t2)

    h2d = statistics.median(t_h2d)
    ker = statistics.median(t_ker)
    d2h = statistics.median(t_d2h)
    return {"h2d": h2d, "kernel": ker, "d2h": d2h, "total": h2d + ker + d2h}, h_out
