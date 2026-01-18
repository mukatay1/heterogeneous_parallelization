#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

// Макрос для проверки ошибок CUD

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

#define N 1000000

// Kernel: поэлементное сложение
__global__ void addArrays(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Замер среднего времени одного kernel (ms) для заданного blockSize
float benchmarkAdd(const float* d_a, const float* d_b, float* d_c, int n, int blockSize, int iters = 300) {
    dim3 block(blockSize);
    dim3 grid((n + blockSize - 1) / blockSize);

    // Warm-up (прогрев контекста и частот GPU)
    addArrays<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        addArrays<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Среднее время одного kernel
    return ms / iters;
}

int main() {
    size_t bytes = N * sizeof(float);

    // Host arrays
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Device arrays
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Поднимаем runtime (важно, чтобы первое измерение не было "огромным")
    CHECK_CUDA(cudaFree(0));

    std::cout << "Task 4: Find optimal block/grid configuration (Task 2 kernel)\n";
    std::cout << "Array size: " << N << "\n";
    std::cout << "-------------------------------------------------------------\n";

    // Кандидаты block size (обычно разумные значения)
    int candidates[] = {64, 128, 256, 512, 1024};

    // 1) Поиск лучшего blockSize
    int bestBlock = candidates[0];
    float bestTime = 1e9f;

    std::cout << "Benchmarking candidates:\n";
    for (int bs : candidates) {
        float t = benchmarkAdd(d_a, d_b, d_c, N, bs, 300);
        std::cout << "  blockSize=" << bs << " -> " << t << " ms (avg per kernel)\n";

        if (t < bestTime) {
            bestTime = t;
            bestBlock = bs;
        }
    }

    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Best (optimized) blockSize = " << bestBlock
              << " with time " << bestTime << " ms\n";

    // 2) Сравнение неоптимальной и оптимизированной
    // Неоптимальную возьмем, например, 64 (часто хуже загрузка GPU)
    int badBlock = 64;

    float badTime = benchmarkAdd(d_a, d_b, d_c, N, badBlock, 300);
    float optTime = benchmarkAdd(d_a, d_b, d_c, N, bestBlock, 300);

    dim3 badGrid((N + badBlock - 1) / badBlock);
    dim3 optGrid((N + bestBlock - 1) / bestBlock);

    std::cout << "\nComparison:\n";
    std::cout << "  Non-optimal config: grid=" << badGrid.x << ", block=" << badBlock
              << " -> " << badTime << " ms\n";
    std::cout << "  Optimized config:   grid=" << optGrid.x << ", block=" << bestBlock
              << " -> " << optTime << " ms\n";
    std::cout << "  Speedup: " << (badTime / optTime) << "x\n";

    // 3) Проверка корректности результата
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    std::cout << "Check C[0] (1.0+2.0) = " << h_c[0] << " (expected 3.0)\n";

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return 0;
}
