# Procesamiento digital de imágenes en paralelo: CPU vs GPU (CUDA)

Proyecto 2. Se aplica un pipeline de procesamiento de imágenes (escala de grises, desenfoque gaussiano y Sobel) a radiografías de tórax y se compara el desempeño de tres versiones del mismo algoritmo:

1. **CPU secuencial**: un solo hilo.
2. **CPU multihilo**: el lote de imágenes se reparte entre 2, 4 y 8 hilos.
3. **GPU con CUDA**: un hilo de GPU por píxel, midiendo por separado la transferencia por PCIe y el cómputo.

Todo está escrito en Python con **Numba** (CPU y CUDA). Las tres versiones usan aritmética entera, así que producen **exactamente el mismo resultado** y se validan píxel por píxel.

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), de Paul Mooney en Kaggle (datos originales de Kermany et al., 2018). 5,856 radiografías en JPEG.

**Autores:** _completa aquí tus datos y los del equipo_

---

## 1. Requisitos

- Windows 10/11 (también funciona en Linux con ajustes menores).
- **Python 3.10 o superior** (probado con 3.13).
- **GPU NVIDIA** con un driver reciente. Para comprobarlo, abre PowerShell y ejecuta `nvidia-smi`: debe mostrar el nombre de tu GPU y un campo `CUDA Version` (con 12.x o 13.x funciona).
- Unos **2 GB de RAM libres** para el lote de 1,500 imágenes y unos **3 GB de VRAM** (la configuración por defecto usa ~2.3 GiB en una GPU de 4 GB).
- Git.
- Conexión a internet para instalar las librerías y descargar el dataset (~1 GB).

> No necesitas instalar el CUDA Toolkit completo: las librerías de CUDA se instalan por `pip`.

---

## 2. Descargar el repositorio

```powershell
git clone <URL-DEL-REPOSITORIO>
cd proyecto_cpd_img
```

El repositorio **no incluye el dataset** (pesa ~1 GB). Se descarga aparte (sección 4).

---

## 3. Instalar las librerías

### Opción A: PyCharm

1. Abre la carpeta del proyecto en PyCharm.
2. Ve a **File → Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Virtualenv Environment → New**. Deja la ubicación en `.venv` y acepta.
3. Abre la pestaña **Terminal** de PyCharm. Debe verse `(.venv)` al inicio de la línea.
4. Instala las dependencias:

```powershell
python -m pip install -r requirements.txt
```

### Opción B: terminal

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell marca un error de permisos al activar el entorno, ejecuta antes:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Contenido recomendado de `requirements.txt`

```
numpy<2.4
numba
numba-cuda[cu13]
pillow
matplotlib
```

- `numpy<2.4` es necesario: `numba-cuda` usa `np.row_stack`, que NumPy eliminó en la versión 2.4. Sin este límite, el primer kernel CUDA falla con `AttributeError: module 'numpy' has no attribute 'row_stack'`.
- `numba-cuda[cu13]` trae las librerías de CUDA 13. Si tu driver solo llega a CUDA 12 (según `nvidia-smi`), usa `numba-cuda[cu12]`.

### Verificar que Python ve la GPU

```powershell
python test_cuda.py
```

Debe imprimir `CUDA disponible: True` y listar tu GPU. Si imprime `False`, revisa la sección de problemas frecuentes.

---

## 4. Descargar y acomodar el dataset

1. Descarga el dataset desde Kaggle (necesitas una cuenta gratuita): <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>
2. Descomprime el zip dentro de `data/` para que quede así:

```
data/
└── dataset_xray/
    └── chest_xray/
        ├── train/   (NORMAL/  PNEUMONIA/)
        ├── test/    (NORMAL/  PNEUMONIA/)
        └── val/     (NORMAL/  PNEUMONIA/)
```

3. **Limpia el zip de Kaggle.** Suele traer una segunda carpeta `chest_xray` repetida dentro de la primera, y una carpeta `__MACOSX`. Bórralas: si se quedan, las imágenes se contarían dos veces.
4. Verifica la carga:

```powershell
python src/verificar_datos.py
```

Debe mostrar **5,856 imágenes**: `train/NORMAL` 1,341 · `train/PNEUMONIA` 3,875 · `test/NORMAL` 234 · `test/PNEUMONIA` 390 · `val/NORMAL` 8 · `val/PNEUMONIA` 8.

> Si el nombre de tu carpeta es distinto, ajusta la constante `RAIZ_DATOS` en `src/cargar_datos.py`.

---

## 5. Estructura del proyecto

```
proyecto_cpd_img/
├── data/                    # dataset (NO se sube a git)
├── resultados/              # CSV, gráficas y tablas generadas
├── src/
│   ├── cargar_datos.py      # lee el lote de imágenes en memoria (RGB, 512x512)
│   ├── filtros_cpu.py       # versión 1: grises, gauss 5x5, Sobel (secuencial)
│   ├── filtros_hilos.py     # versión 2: reparto del lote entre hilos
│   ├── filtros_cuda.py      # versión 3: kernels CUDA + medición de PCIe
│   ├── benchmark.py         # mide las tres versiones y guarda el CSV
│   ├── graficar.py          # genera gráficas y tablas desde el CSV
│   └── verificar_datos.py   # cuenta y revisa las imágenes del dataset
├── main.py                  # ejecuta el benchmark completo
├── hacer_graficas.py        # regenera gráficas y tablas desde el CSV
├── test_cuda.py             # comprueba que Numba ve la GPU
├── test_carga.py            # prueba la carga del lote
├── test_filtros.py          # revisión visual de los filtros en CPU
├── test_hilos.py            # valida y mide la versión multihilo
├── test_gpu.py              # valida y mide la versión CUDA
└── requirements.txt
```

Los archivos `test_*.py` son pruebas por etapa: sirven para comprobar cada versión por separado antes de correr el benchmark completo.

---

## 6. Ejecutar

Todos los comandos se lanzan **desde la raíz del proyecto**, con el entorno virtual activo.

**Pruebas por etapa (opcionales, para comprobar que todo funciona):**

```powershell
python test_carga.py      # carga 100 imágenes: debe dar (100, 512, 512, 3)
python test_filtros.py    # genera resultados/ejemplo_filtros.png
python test_hilos.py      # tabla secuencial vs 2, 4 y 8 hilos (columna Igual? = True)
python test_gpu.py        # valida CUDA vs CPU (diferencia máxima 0) y mide tiempos
```

**Benchmark completo** (tarda unos 2 minutos):

```powershell
python main.py
```

Guarda `resultados/benchmark.csv`. Después, genera las gráficas y tablas:

```powershell
python hacer_graficas.py
```

### Parámetros que puedes cambiar (en `main.py`)

| Constante | Valor por defecto | Significado |
|---|---|---|
| `TAMANOS` | `[100, 250, 500, 1000, 1500]` | tamaños de problema (número de imágenes) |
| `HILOS` | `[2, 4, 8]` | configuraciones de hilos de CPU |
| `REPS` | `5` | repeticiones por medición (se usa la mediana) |

El tamaño de imagen (512×512) se define con el parámetro `tam` de `cargar_lote` en `src/cargar_datos.py`.

> Memoria de GPU: cada imagen de 512×512 necesita ~1.5 MiB de VRAM. Con una GPU de 4 GB, 1,500 imágenes (~2.3 GiB) caben con margen. Con 2,000 ya es arriesgado. El benchmark revisa la VRAM libre y omite la GPU en el tamaño que no quepa, en lugar de fallar.

---

## 7. Resultados generados (`resultados/`)

| Archivo | Contenido |
|---|---|
| `benchmark.csv` | todos los tiempos y aceleraciones, una fila por tamaño de problema |
| `grafica_aceleracion.png` | tamaño del problema contra aceleración (CPU 2/4/8 hilos, GPU pura y GPU real) |
| `grafica_tiempos.png` | tamaño del problema contra tiempo de cada versión |
| `grafica_desglose_gpu.png` | de qué se compone el tiempo de la GPU: transferencia H2D, kernels y D2H |
| `tabla_tiempos.md` | tabla comparativa de tiempos: secuencial, multihilo, PCIe, GPU y total GPU |
| `tabla_aceleracion.md` | aceleración pura y real, y el porcentaje del tiempo que consume PCIe |
| `ejemplo_filtros.png` | una radiografía en gris, desenfocada y con Sobel |

Las gráficas de aceleración y tiempo usan escala logarítmica, porque entre el secuencial y la GPU hay más de un orden de magnitud de diferencia.

---

## 8. Cómo interpretar los resultados

### Columnas del CSV (tiempos en segundos)

| Columna | Significado |
|---|---|
| `tamano` | número de imágenes del lote |
| `t_secuencial` | CPU con un hilo (línea base) |
| `t_hilos_2`, `t_hilos_4`, `t_hilos_8` | CPU con esa cantidad de hilos |
| `t_h2d` | transferencia RAM → VRAM por PCIe (Host to Device) |
| `t_gpu` | solo el cómputo de los tres kernels en la GPU |
| `t_d2h` | transferencia VRAM → RAM por PCIe (Device to Host) |
| `t_total_gpu` | `t_h2d + t_gpu + t_d2h` |
| `acel_hilos_N` | `t_secuencial / t_hilos_N` |
| `acel_pura` | `t_secuencial / t_gpu` (ignora PCIe) |
| `acel_real` | `t_secuencial / t_total_gpu` (incluye PCIe) |

Si `t_gpu` y sus columnas dicen `nan`, ese tamaño no cupo en la VRAM y se omitió la GPU.

### Qué significa cada métrica

- **Aceleración (speedup)** = `T_secuencial / T_versión`. Un 4x significa que tarda la cuarta parte.
- **Eficiencia** (multihilo) = aceleración ÷ número de hilos. Un 100 % sería escalamiento perfecto. Normalmente baja al subir los hilos por el ancho de banda de memoria, la caché y porque hilos lógicos (hyperthreading) comparten núcleo físico.
- **Aceleración pura** mide solo el cómputo de la GPU. Es la mejor cifra posible, pero no es lo que experimenta un programa real.
- **Aceleración real con PCIe** incluye copiar los datos a la GPU y traerlos de vuelta. Es la que importa en la práctica.
- **PCIe como % del total GPU**: cuánto del tiempo de la GPU se va en mover datos. Si es alto (más de 50 %), el cuello de botella no es el cómputo sino la transferencia.

### Ejemplo de lectura

Corrida de prueba con 500 imágenes de 512×512 en una RTX 3050 Laptop (4 GB):

| Versión | Tiempo | Aceleración |
|---|---|---|
| Secuencial | 1.63 s | 1.00x |
| 8 hilos | 0.52 s | 3.11x |
| Transferencia H2D | 0.059 s | |
| GPU (kernels) | 0.058 s | 28.2x (pura) |
| Transferencia D2H | 0.025 s | |
| GPU total con PCIe | 0.142 s | 11.5x (real) |

En este ejemplo, la GPU es 28x más rápida si solo se cuenta el cómputo, pero al incluir PCIe baja a 11.5x: el 59 % del tiempo de GPU se va en transferir datos. Aun así, sigue siendo unas 3.7 veces más rápida que 8 hilos de CPU. _Tus números serán distintos._

### Cosas a observar al redactar las conclusiones

- **Efecto del tamaño del problema.** Con pocas imágenes, los costos fijos (lanzar hilos, arrancar kernels) pesan más y la aceleración es menor. Con lotes grandes la curva se estabiliza. Compara cómo cambia cada curva entre 100 y 1,500 imágenes.
- **Multihilo contra GPU.** Fíjate en cuántas veces más rápida es la GPU que la mejor configuración de hilos, tanto en versión pura como real.
- **Peso de PCIe.** Observa cómo cambia el porcentaje de transferencia al crecer el lote (gráfica de desglose).
- **Dónde deja de escalar el multihilo.** Compara la eficiencia de 2, 4 y 8 hilos con los núcleos físicos y lógicos de tu CPU.

---

## 9. Detalles de implementación

- **Pipeline por imagen:** RGB → escala de grises (`(77R + 150G + 29B) / 256`) → gaussiano 5×5 (pesos binomiales `[1 4 6 4 1] × [1 4 6 4 1]`, suma 256) → Sobel (magnitud `√(Gx² + Gy²)` recortada a 255). Los bordes repiten el píxel más cercano.
- **Aritmética entera:** garantiza que CPU, hilos y GPU den el mismo resultado exacto (diferencia máxima 0).
- **Multihilo:** el lote se divide en rangos de imágenes, uno por hilo. Cada hilo escribe en su región del arreglo de salida, así que no hay condiciones de carrera ni se necesitan locks. Las funciones usan `nogil=True`, por lo que los hilos de Python corren en paralelo real.
- **CUDA:** un hilo por píxel de salida, bloques de 16×16, malla 3D (x = columnas, y = filas, z = imagen del lote). Tres kernels (gris, gauss, Sobel) en el mismo stream.
- **Medición:**
  - Las imágenes se cargan en memoria **antes** de medir, para no contar la lectura de disco.
  - Hay una corrida de calentamiento para absorber la compilación JIT y el arranque de CUDA.
  - Se usa la **mediana** de varias repeticiones, que resiste los picos de ruido del sistema.
  - En la GPU se llama a `cuda.synchronize()` antes y después de cada tramo, porque las llamadas a CUDA son asíncronas.
  - Los búferes de la GPU se reservan una sola vez antes de medir.
- **Redimensionado:** todas las imágenes se estiran a 512×512 para poder procesarlas en un solo arreglo. Esto deforma las radiografías que no son cuadradas; no afecta la comparación de rendimiento, que es el objetivo del proyecto.

### Recomendaciones para mediciones confiables

- Conecta el laptop al cargador y usa el modo de energía de máximo rendimiento.
- Cierra el navegador y otros programas pesados.
- Si un tiempo se sale de la tendencia, repite toda la corrida. En un laptop, la temperatura y el ahorro de energía pueden variar los resultados entre corridas.

---

## 10. Problemas frecuentes

| Problema | Causa y solución |
|---|---|
| `AttributeError: module 'numpy' has no attribute 'row_stack'` al lanzar un kernel | NumPy es demasiado nuevo para `numba-cuda`. Ejecuta `pip install "numpy<2.4"`. |
| `cuda.is_available()` da `False` | Falta el driver de NVIDIA o está desactualizado, o el equipo usa solo la gráfica integrada. Revisa `nvidia-smi` y reinstala/actualiza el driver. Comprueba también que instalaste `numba-cuda[cu13]` (o `[cu12]` según tu driver). |
| `FileNotFoundError` o `No encontré imágenes` | La ruta del dataset no coincide. Revisa que exista `data/dataset_xray/chest_xray/train` y ajusta `RAIZ_DATOS` en `src/cargar_datos.py` si usaste otro nombre. |
| El total de imágenes es ~11,700 | Quedó la carpeta `chest_xray` duplicada dentro de la otra. Borra la repetida. |
| `ModuleNotFoundError: No module named 'src'` | Falta `src/__init__.py` (archivo vacío), o ejecutas el script desde otra carpeta. Corre los comandos desde la raíz del proyecto. |
| `FileNotFoundError: ...benchmark.csv` en `hacer_graficas.py` | Aún no hay CSV: ejecuta primero `python main.py`. Además, `hacer_graficas.py` debe estar en la raíz del proyecto, no dentro de `src/`. |
| Error de memoria de la GPU | Reduce el tamaño máximo en `TAMANOS` (por ejemplo, quita 1500) o baja el tamaño de imagen. |
| La primera ejecución tarda más | Numba compila el código la primera vez. Las siguientes arrancan más rápido (`cache=True`). |
| PowerShell no deja activar el venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## 11. Ficha técnica del equipo de pruebas

_Completa esta sección con los datos de tu equipo para el documento._

| Componente | Detalle |
|---|---|
| CPU | _modelo, núcleos físicos / hilos lógicos (Administrador de tareas → Rendimiento → CPU)_ |
| RAM | _GB_ |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB de VRAM |
| Driver NVIDIA | 592.00 (CUDA 13.1) |
| Sistema operativo | _Windows 10/11, versión_ |
| Python / Numba / NumPy | _versiones: `pip list`_ |

Para la GPU puedes obtener más datos (multiprocesadores, compute capability) con:

```python
from numba import cuda
d = cuda.get_current_device()
print(d.name, d.compute_capability, d.MULTIPROCESSOR_COUNT)
```

---

## 12. Subir cambios al repositorio

El archivo `.gitignore` debe excluir como mínimo:

```
data/
.venv/
__pycache__/
.idea/
```

Los resultados (`resultados/`) sí conviene subirlos, para que las gráficas y tablas se vean en el repositorio.

```powershell
git add .
git commit -m "Descripción del cambio"
git push
```