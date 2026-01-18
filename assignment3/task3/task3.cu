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

// 1) Коалесцированный доступ
// Поток idx читает/пишет data[idx] -> соседние потоки в warp идут по соседним адресам
__global__ void coalescedKernel(float* data, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scalar;
    }
}

// 2) Некоалесцированный доступ
// Поток idx обращается к "перемешанному" индексу: (idx * stride) % n
// При большом stride в пределах warp адреса будут далеко друг от друга
__global__ void nonCoalescedKernel(float* data, float scalar, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int j = (idx * stride) % n;     // разброс по памяти
        data[j] = data[j] * scalar;
    }
}

// Функция замера среднего времени kernel (в ms) через cudaEvent
float benchmarkKernel(void (*kernel)(float*, float, int), float* d_data, float scalar, int n,
                      dim3 grid, dim3 block, int iters) {
    // Этот helper не используем для nonCoalesced (там другой сигнатуры kernel)
    cudaEvent_t start, stop;
    // Warm-up
    ((void(*)(float*, float, int))kernel)<<<grid, block>>>(d_data, scalar, n);
    for (int i = 0; i < iters; ++i) {
        ((void(*)(float*, float, int))kernel)<<<grid, block>>>(d_data, scalar, n);
    }

    float ms = 0.0f;

    return ms / iters;
}

// Функция замера среднего времени kernel coalesced
float benchmarkNonCoalesced(float* d_data, float scalar, int n, int stride,
                            dim3 grid, dim3 block, int iters) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    nonCoalescedKernel<<<grid, block>>>(d_data, scalar, n, stride);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        nonCoalescedKernel<<<grid, block>>>(d_data, scalar, n, stride);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / iters;
}

int main() {
    const size_t bytes = N * sizeof(float);
    const float scalar = 1.000001f; // маленький множитель, чтобы не "взрывать" значения
    const int blockSize = 256;
    const dim3 block(blockSize);
    const dim3 grid((N + blockSize - 1) / blockSize);

    // Host data
    std::vector<float> h_data(N, 1.0f);

    // Device buffers (раздельно для честного сравнения)
    float *d_coal = nullptr, *d_non = nullptr;
    CHECK_CUDA(cudaMalloc(&d_coal, bytes));
    CHECK_CUDA(cudaMalloc(&d_non, bytes));

    CHECK_CUDA(cudaMemcpy(d_coal, h_data.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_non,  h_data.data(), bytes, cudaMemcpyHostToDevice));

    // Прогрев runtime
    CHECK_CUDA(cudaFree(0));

    // stride выбираем большим и "неудобным"
    // желательно взаимно простым с N, чтобы проход был "перемешанный"
    const int stride = 9973;

    const int iters = 200;

    std::cout << "Task 3: Coalesced vs Non-coalesced global memory access\n";
    std::cout << "Array size: " << N << "\n";
    std::cout << "Block size: " << blockSize << "\n";
    std::cout << "Stride (non-coalesced): " << stride << "\n";
    std::cout << "------------------------------------------------------\n";

    float tCoal = benchmarkKernel((void(*)(float*, float, int))coalescedKernel,
                                  d_coal, scalar, N, grid, block, iters);

    float tNon  = benchmarkNonCoalesced(d_non, scalar, N, stride, grid, block, iters);

    std::cout << "1. Coalesced Kernel:     " << tCoal << " ms (avg)\n";
    std::cout << "2. Non-coalesced Kernel: " << tNon  << " ms (avg)\n";
    std::cout << "Speedup (coal/non):      " << (tNon / tCoal) << "x\n";
    std::cout << "------------------------------------------------------\n";

    // Проверка корректности: скопируем пару элементов
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_coal, bytes, cudaMemcpyDeviceToHost));
    std::cout << "Check (coalesced) h_data[0]   = " << h_data[0] << "\n";
    std::cout << "Check (coalesced) h_data[n-1] = " << h_data[N-1] << "\n";

    CHECK_CUDA(cudaFree(d_coal));
    CHECK_CUDA(cudaFree(d_non));
    return 0;
}
