#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// (a) Только глобальная память: atomicAdd на каждый элемент (намеренно медленно)
__global__ void reduce_global_atomic(const float* __restrict__ d_in, float* d_out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(d_out, d_in[i]);
}

// (b) Global + Shared: редукция в shared внутри блока + один atomicAdd на блок
__global__ void reduce_shared_block(const float* __restrict__ d_in, float* d_out, size_t n) {
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t i   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? d_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_out, sdata[0]);
}

static float time_kernel_global_atomic(const float* d_in, size_t n, int threads) {
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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
    return ms;
}

static float time_kernel_shared(const float* d_in, size_t n, int threads) {
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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
    return ms;
}

static void fill_random(std::vector<float>& h, uint32_t seed=12345) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : h) x = dist(rng);
}

int main() {
    const int threads = 256;
    const std::vector<size_t> sizes = {10'000, 100'000, 1'000'000};
    const int WARMUP = 1;
    const int REPEATS = 10; // усреднение для стабильности

    std::ofstream csv("times.csv");
    csv << "N,global_atomic_ms,shared_reduce_ms\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "N\tglobal_atomic(ms)\tshared_reduce(ms)\n";

    for (size_t N : sizes) {
        // Host data
        std::vector<float> h(N);
        fill_random(h);

        // Device data (global memory)
        float* d_in = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            (void)time_kernel_global_atomic(d_in, N, threads);
            (void)time_kernel_shared(d_in, N, threads);
        }

        // Measure (average)
        double tA = 0.0, tB = 0.0;
        for (int i = 0; i < REPEATS; ++i) tA += time_kernel_global_atomic(d_in, N, threads);
        for (int i = 0; i < REPEATS; ++i) tB += time_kernel_shared(d_in, N, threads);
        tA /= REPEATS;
        tB /= REPEATS;

        std::cout << N << "\t" << tA << "\t\t\t" << tB << "\n";
        csv << N << "," << tA << "," << tB << "\n";

        CUDA_CHECK(cudaFree(d_in));
    }

    csv.close();
    std::cout << "\nSaved: times.csv\n";
    return 0;
}
