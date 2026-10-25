from numba import cuda

print("CUDA disponible:", cuda.is_available())
cuda.detect()