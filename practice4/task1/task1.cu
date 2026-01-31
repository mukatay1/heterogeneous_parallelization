// task1_generate_data.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>   // std::accumulate

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

int main() {
    const size_t N = 1'000'000;             // размер массива
    const size_t bytes = N * sizeof(float); // будем хранить float

    // 1) Генерация на CPU
    std::vector<float> h(N);

    std::mt19937 rng(12345); // фиксируем seed для воспроизводимости
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        h[i] = dist(rng);
    }

    // (необязательно) быстрая проверка
    double cpu_sum = std::accumulate(h.begin(), h.end(), 0.0);
    std::cout << "CPU check sum = " << cpu_sum << "\n";

    // 2) Выделение памяти на GPU (глобальная память)
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));

    // 3) Копирование данных CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << "Generated " << N << " random floats and copied to GPU.\n";

    // 4) Освобождение памяти
    CUDA_CHECK(cudaFree(d));
    return 0;
}
