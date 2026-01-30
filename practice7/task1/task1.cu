#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#define BLOCK_SIZE 256

int main() {
    const int N = 1024; // тестовый размер массива
    const size_t bytes = N * sizeof(float);

    // ===== Инициализация данных =====
    std::vector<float> h_input(N);
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // чтобы сумма была легко проверяема
    }

    // ===== CPU сумма (для проверки) =====
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_sum = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // ===== Выделение памяти на GPU =====
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_output, gridSize * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // ===== Запуск ядра =====
    auto gpu_start = std::chrono::high_resolution_clock::now();

    reduce_sum<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input, d_output, N
    );

    cudaDeviceSynchronize();

    // ===== Копируем частичные суммы =====
    std::vector<float> h_output(gridSize);
    cudaMemcpy(h_output.data(), d_output,
               gridSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Финальная редукция на CPU
    float gpu_sum = std::accumulate(h_output.begin(), h_output.end(), 0.0f);

    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;

    // ===== Результаты =====
    std::cout << "===== Task 1: CUDA Reduction =====\n\n";
    std::cout << "Размер массива: " << N << "\n";
    std::cout << "CPU сумма: " << cpu_sum << "\n";
    std::cout << "GPU сумма: " << gpu_sum << "\n\n";

    std::cout << "CPU time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU time: " << gpu_time.count() << " ms\n";

    if (std::abs(cpu_sum - gpu_sum) < 1e-5)
        std::cout << "\nРезультаты совпадают ✔\n";
    else
        std::cout << "\nОшибка вычислений ✘\n";

    // ===== Очистка =====
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
