// task1_sum_global.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#define CUDA_CHECK(call) do {                             \
    cudaError_t err = (call);                             \
    if (err != cudaSuccess) {                             \
        std::cerr << "CUDA error: "                       \
                  << cudaGetErrorString(err)              \
                  << " at " << __FILE__ << ":" << __LINE__\
                  << std::endl;                           \
        std::exit(1);                                     \
    }                                                     \
} while(0)

// CUDA-ядро для суммирования элементов массива.
// Каждый поток проходит по массиву с шагом stride
// и накапливает частичную сумму, которая затем
// добавляется в общий результат через atomicAdd.
__global__ void sum_global_atomic(const float* a, int n, float* out_sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += stride) {
        local += a[i];
    }

    atomicAdd(out_sum, local);
}

// Последовательное суммирование на CPU.
// Используется для проверки корректности
// и сравнения времени выполнения.
double cpu_sum(const std::vector<float>& a) {
    double s = 0.0;
    for (float x : a) {
        s += x;
    }
    return s;
}

int main() {
    const int N = 100000;

    // Формирование входного массива
    std::vector<float> h_a(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        h_a[i] = dist(rng);
    }

    // Запуск последовательной версии на CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    double h_sum_cpu = cpu_sum(h_a);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Выделение памяти на GPU
    float* d_a = nullptr;
    float* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

    // Копирование данных на устройство
    CUDA_CHECK(cudaMemcpy(d_a,
                          h_a.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Обнуление результирующей суммы
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    // Настройка параметров запуска ядра
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 256) {
        grid = 256;
    }

    // Измерение времени выполнения CUDA-ядра
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    sum_global_atomic<<<grid, block>>>(d_a, N, d_sum);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // Копирование результата обратно на CPU
    float h_sum_gpu = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum_gpu,
                          d_sum,
                          sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Проверка расхождения результатов
    double diff = std::abs(h_sum_cpu - static_cast<double>(h_sum_gpu));

    std::cout << "N = " << N << "\n";
    std::cout << "CPU sum = " << h_sum_cpu << "\n";
    std::cout << "GPU sum = " << h_sum_gpu << "\n";
    std::cout << "Absolute difference = " << diff << "\n";
    std::cout << "CPU time (ms) = " << cpu_ms << "\n";
    std::cout << "GPU kernel time (ms) = " << gpu_ms << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_sum));

    return 0;
}
