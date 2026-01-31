// task2_reduction.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// ------------------------
// (a) Только глобальная память (намеренно неэффективно):
// каждый поток делает atomicAdd в глобальную сумму.
// ------------------------
__global__ void reduce_global_atomic(const float* __restrict__ d_in, float* d_out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(d_out, d_in[i]);
    }
}

// ------------------------
// (b) Глобальная + разделяемая память:
// редукция суммы внутри блока в shared, затем один atomicAdd на блок.
// ------------------------
__global__ void reduce_shared_block(const float* __restrict__ d_in, float* d_out, size_t n) {
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t i   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    float x = (i < n) ? d_in[i] : 0.0f;
    sdata[tid] = x;
    __syncthreads();

    // редукция в shared (power-of-two)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_out, sdata[0]);
    }
}

static float run_and_time_global_atomic(const float* d_in, size_t n, int threads, float& gpu_sum_out) {
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    int blocks = (int)((n + threads - 1) / threads);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    reduce_global_atomic<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(&gpu_sum_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));

    return ms;
}

static float run_and_time_shared_block(const float* d_in, size_t n, int threads, float& gpu_sum_out) {
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    int blocks = (int)((n + threads - 1) / threads);
    size_t shmem = (size_t)threads * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    reduce_shared_block<<<blocks, threads, shmem>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(&gpu_sum_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));

    return ms;
}

int main() {
    const size_t N = 1'000'000;
    const int threads = 256;

    // --- Генерация данных на CPU
    std::vector<float> h(N);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) h[i] = dist(rng);

    // CPU контроль
    double cpu_sum = std::accumulate(h.begin(), h.end(), 0.0);
    std::cout << "CPU sum = " << cpu_sum << "\n";

    // --- Копирование на GPU (глобальная память)
    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // --- (a) только глобальная память
    float gpu_sum_a = 0.0f;
    float ms_a = run_and_time_global_atomic(d_in, N, threads, gpu_sum_a);

    // --- (b) global + shared
    float gpu_sum_b = 0.0f;
    float ms_b = run_and_time_shared_block(d_in, N, threads, gpu_sum_b);

    // --- Вывод
    std::cout << "\n=== Results ===\n";
    std::cout << "(a) Global-only (atomic each element)\n";
    std::cout << "  GPU sum = " << gpu_sum_a << ", time = " << ms_a << " ms\n";

    std::cout << "(b) Global+Shared (block reduce + atomic per block)\n";
    std::cout << "  GPU sum = " << gpu_sum_b << ", time = " << ms_b << " ms\n";

    // Погрешность (float vs double)
    auto rel_err = [&](double ref, double val) {
        return std::abs(ref - val) / (std::abs(ref) + 1e-9);
    };
    std::cout << "\nRelative error vs CPU:\n";
    std::cout << "  (a) " << rel_err(cpu_sum, gpu_sum_a) << "\n";
    std::cout << "  (b) " << rel_err(cpu_sum, gpu_sum_b) << "\n";

    CUDA_CHECK(cudaFree(d_in));
    return 0;
}
