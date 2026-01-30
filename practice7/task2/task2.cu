#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#define BLOCK_SIZE 1024   // Один блок, чтобы упростить реализацию

// CUDA-ядро префиксной суммы (inclusive scan)
// Реализация Hillis–Steele с использованием shared memory
__global__ void prefix_scan(const float* input, float* output, int n) {
    // Разделяемая память — быстрая память внутри блока
    extern __shared__ float sdata[];

    int tid = threadIdx.x;   // локальный индекс потока
    int i = tid;             // глобальный индекс (1 блок)

    // 1. Загрузка данных из глобальной памяти в shared memory
    if (i < n)
        sdata[tid] = input[i];
    else
        sdata[tid] = 0.0f;

    __syncthreads(); // ждем, пока все потоки загрузят данные

    // 2. Итеративное накопление (Hillis–Steele scan)
    // offset = 1, 2, 4, 8, ...
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float value = 0.0f;

        if (tid >= offset)
            value = sdata[tid - offset];

        __syncthreads();
        sdata[tid] += value;
        __syncthreads();
    }

    // 3. Запись результата обратно в глобальную память
    if (i < n)
        output[i] = sdata[tid];
}

int main() {
    const int N = 1024;                    // Размер массива
    const size_t bytes = N * sizeof(float);

    // Инициализация входных данных
    std::vector<float> h_input(N);
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // Для простоты: [1,1,1,...]
    }

    std::vector<float> h_output(N);

    // CPU версия (для проверки)
    auto cpu_start = std::chrono::high_resolution_clock::now();

    std::vector<float> cpu_scan(N);
    cpu_scan[0] = h_input[0];
    for (int i = 1; i < N; i++) {
        cpu_scan[i] = cpu_scan[i - 1] + h_input[i];
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // Выделение памяти на GPU
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Копирование данных Host → Device
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // Запуск CUDA-ядра
    auto gpu_start = std::chrono::high_resolution_clock::now();

    prefix_scan<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input, d_output, N
    );

    cudaDeviceSynchronize();

    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;

    // Копирование результата Device → Host
    cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    // Проверка корректности
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (abs(h_output[i] - cpu_scan[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    // Вывод результатов
    std::cout << "===== Task 2: CUDA Prefix Sum (Scan) =====\n\n";
    std::cout << "Размер массива: " << N << "\n";

    std::cout << "CPU time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU time: " << gpu_time.count() << " ms\n\n";

    std::cout << "Проверка корректности: "
              << (correct ? "УСПЕХ ✔" : "ОШИБКА ✘") << "\n\n";

    std::cout << "Пример результата:\n";
    std::cout << "input[0] = " << h_input[0]
              << ", scan[0] = " << h_output[0] << "\n";
    std::cout << "input[N-1] = " << h_input[N - 1]
              << ", scan[N-1] = " << h_output[N - 1] << "\n";

    // Освобождение памяти
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
