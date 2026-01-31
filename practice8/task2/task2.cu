#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

#define N 1000000   // Размер массива

// ---------------- CUDA ядро ----------------
// Каждому элементу массива умножаем значение на 2
__global__ void gpu_process(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] *= 2.0f;
    }
}

// ---------------- CPU функция (OpenMP) ----------------
void cpu_process(float* data, int size)
{
    // Распараллеливаем цикл с помощью OpenMP
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        data[i] *= 2.0f;
    }
}

int main()
{
    // Выделение памяти на CPU
    float* h_data = new float[N];

    // Инициализация массива
    for (int i = 0; i < N; i++)
    {
        h_data[i] = 1.0f;
    }

    // Делим массив на две части
    int cpu_size = N / 2;
    int gpu_size = N - cpu_size;

    // ---------------- GPU память ----------------
    float* d_data;
    cudaMalloc((void**)&d_data, gpu_size * sizeof(float));

    // Копируем вторую половину массива на GPU
    cudaMemcpy(d_data,
               h_data + cpu_size,
               gpu_size * sizeof(float),
               cudaMemcpyHostToDevice);

    // ---------------- Замер времени ----------------
    double start_time = omp_get_wtime();

    // --------- CPU вычисления ---------
    cpu_process(h_data, cpu_size);

    // --------- GPU вычисления ---------
    int threadsPerBlock = 256;
    int blocksPerGrid = (gpu_size + threadsPerBlock - 1) / threadsPerBlock;

    gpu_process<<<blocksPerGrid, threadsPerBlock>>>(d_data, gpu_size);
    cudaDeviceSynchronize();

    // Копируем результат обратно на CPU
    cudaMemcpy(h_data + cpu_size,
               d_data,
               gpu_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    double end_time = omp_get_wtime();

    // ---------------- Результаты ----------------
    std::cout << "Гибридное время выполнения: "
              << end_time - start_time
              << " секунд" << std::endl;

    // Проверка корректности
    std::cout << "Пример значений массива:" << std::endl;
    std::cout << "h_data[0] = " << h_data[0] << std::endl;
    std::cout << "h_data[N/2] = " << h_data[N/2] << std::endl;

    // Освобождение памяти
    delete[] h_data;
    cudaFree(d_data);

    return 0;
}
