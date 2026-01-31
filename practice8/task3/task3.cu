#include <iostream>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

#define N 1000000  // Размер массива

// ---------------- CUDA ядро ----------------
// Умножаем каждый элемент на 2 (GPU часть)
__global__ void gpu_mul2(float* data, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] *= 2.0f;
}

// ---------------- CPU функция (OpenMP) ----------------
// Умножаем каждый элемент на 2 (CPU часть)
void cpu_mul2(float* data, int size)
{
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
        data[i] *= 2.0f;
}

int main()
{
    // 1) Создаем массив на CPU и инициализируем
    std::vector<float> h(N, 1.0f);

    // 2) Делим массив на две части:
    //    первая половина -> CPU, вторая половина -> GPU
    int cpu_size = N / 2;
    int gpu_size = N - cpu_size;

    // 3) Выделяем память на GPU под вторую половину
    float* d = nullptr;
    cudaMalloc((void**)&d, gpu_size * sizeof(float));

    // 4) Копируем вторую половину массива на GPU
    cudaMemcpy(d, h.data() + cpu_size,
               gpu_size * sizeof(float),
               cudaMemcpyHostToDevice);

    // Настройка CUDA запуска
    int threads = 256;
    int blocks = (gpu_size + threads - 1) / threads;

    // 5) Замер общего времени гибридной обработки
    double t0 = omp_get_wtime();

    // 6) Запускаем CPU и GPU "одновременно":
    //    - CPU считает в одном потоке (или в нескольких из-за OpenMP)
    //    - GPU ядро запускается асинхронно (по умолчанию) и выполняется параллельно
    cpu_mul2(h.data(), cpu_size);          // CPU: первая половина
    gpu_mul2<<<blocks, threads>>>(d, gpu_size);  // GPU: вторая половина

    // 7) Дожидаемся завершения GPU
    cudaDeviceSynchronize();

    // 8) Копируем результат GPU обратно во вторую половину массива на CPU
    cudaMemcpy(h.data() + cpu_size, d,
               gpu_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    double t1 = omp_get_wtime();

    // 9) Вывод времени и быстрая проверка корректности
    std::cout << "Гибридное время (CPU+GPU): " << (t1 - t0) << " сек\n";
    std::cout << "Проверка: h[0]=" << h[0]
              << ", h[N/2]=" << h[N/2]
              << ", h[N-1]=" << h[N-1] << "\n";
    // Ожидаем везде 2.0 (т.к. было 1.0 и умножили на 2)

    // 10) Освобождаем память GPU
    cudaFree(d);

    return 0;
}
