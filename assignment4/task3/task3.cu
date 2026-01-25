#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <cmath>

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s (%d): %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        std::exit(1);                                                   \
    }                                                                   \
} while (0)

// GPU-ядро
// Выполняет простую обработку массива:
// out[i] = in[i] * alpha + beta
__global__ void process_kernel(const float* in,
                               float* out,
                               int n,
                               float alpha,
                               float beta) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = in[gid] * alpha + beta;
    }
}

// CPU-функция обработки массива
// Используется как для CPU-only версии,
// так и для CPU-части гибридного варианта
void cpu_process(const float* in,
                 float* out,
                 int n,
                 float alpha,
                 float beta) {
    for (int i = 0; i < n; ++i) {
        out[i] = in[i] * alpha + beta;
    }
}

// Проверка равенства float-значений с погрешностью
bool almost_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps * (1.0f + std::fabs(a) + std::fabs(b));
}

int main() {
    const int N = 1'000'000;     // общий размер массива
    const float alpha = 1.7f;    // коэффициенты обработки
    const float beta  = 3.0f;

    // Делим массив на две части
    const int Ncpu = N / 2;      // первая половина — CPU
    const int Ngpu = N - Ncpu;   // вторая половина — GPU
    const int gpuOffset = Ncpu;

    // Выделение pinned host memory
    // Нужно для корректной асинхронной работы CPU и GPU
    float *h_in, *h_out_cpu, *h_out_gpu, *h_out_hybrid;
    CUDA_CHECK(cudaMallocHost(&h_in,         N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out_cpu,    N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out_gpu,    N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out_hybrid, N * sizeof(float)));

    // Инициализация входных данных
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    for (int i = 0; i < N; ++i) {
        h_in[i] = dist(rng);
    }

    // 1. CPU-only реализация
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_process(h_in, h_out_cpu, N, alpha, beta);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // 2. GPU-only реализация
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    cudaEvent_t gstart, gstop;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gstop);

    cudaEventRecord(gstart);

    // Копирование всего массива на GPU
    CUDA_CHECK(cudaMemcpy(d_in, h_in,
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    int BLOCK = 256;
    int GRID  = (N + BLOCK - 1) / BLOCK;

    // Запуск GPU-ядра
    process_kernel<<<GRID, BLOCK>>>(d_in, d_out, N, alpha, beta);

    // Копирование результата обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaEventRecord(gstop);
    cudaEventSynchronize(gstop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, gstart, gstop);

    // 3. Гибридная реализация (CPU + GPU параллельно)
    float *d_in2, *d_out2;
    CUDA_CHECK(cudaMalloc(&d_in2,  Ngpu * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out2, Ngpu * sizeof(float)));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto hybrid_start = std::chrono::high_resolution_clock::now();

    // CPU обрабатывает первую половину массива
    std::thread cpu_thread([&]() {
        cpu_process(h_in, h_out_hybrid, Ncpu, alpha, beta);
    });

    // GPU обрабатывает вторую половину массива асинхронно
    CUDA_CHECK(cudaMemcpyAsync(d_in2,
                               h_in + gpuOffset,
                               Ngpu * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));

    int GRID2 = (Ngpu + BLOCK - 1) / BLOCK;
    process_kernel<<<GRID2, BLOCK, 0, stream>>>(d_in2, d_out2,
                                                Ngpu, alpha, beta);

    CUDA_CHECK(cudaMemcpyAsync(h_out_hybrid + gpuOffset,
                               d_out2,
                               Ngpu * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               stream));

    // Ожидание завершения CPU и GPU частей
    cpu_thread.join();
    cudaStreamSynchronize(stream);

    auto hybrid_end = std::chrono::high_resolution_clock::now();
    double hybrid_ms =
        std::chrono::duration<double, std::milli>(hybrid_end - hybrid_start).count();

    // Проверка корректности гибридного варианта
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (!almost_equal(h_out_hybrid[i], h_out_cpu[i])) {
            ok = false;
            break;
        }
    }

    // Вывод результатов
    printf("Array size: %d\n", N);
    printf("CPU-only time:    %.3f ms\n", cpu_ms);
    printf("GPU-only time:    %.3f ms\n", gpu_ms);
    printf("Hybrid time:      %.3f ms\n", hybrid_ms);
    printf("Correctness:      %s\n", ok ? "OK" : "FAIL");

    // Освобождение памяти
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_in2);
    cudaFree(d_out2);

    cudaFreeHost(h_in);
    cudaFreeHost(h_out_cpu);
    cudaFreeHost(h_out_gpu);
    cudaFreeHost(h_out_hybrid);

    return 0;
}
