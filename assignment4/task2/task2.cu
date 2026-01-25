#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <random>

// Макрос для проверки ошибок CUDA.
// Если CUDA-вызов завершился с ошибкой — программа аварийно завершается.
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s (%d): %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        std::exit(1);                                                   \
    }                                                                   \
} while (0)

// CPU: последовательная реализация inclusive prefix sum (scan)
// Каждый элемент выходного массива содержит сумму
// всех элементов входного массива до текущего индекса включительно
void cpu_inclusive_scan(const std::vector<int>& in,
                        std::vector<int>& out) {
    out.resize(in.size());
    long long acc = 0;          // аккумулятор суммы
    for (size_t i = 0; i < in.size(); ++i) {
        acc += in[i];           // прибавляем текущий элемент
        out[i] = (int)acc;      // сохраняем результат
    }
}

// GPU: блоковый inclusive scan с использованием shared memory
// Реализован алгоритм Blelloch (upsweep + downsweep)
__global__ void block_inclusive_scan(const int* d_in,
                                     int* d_out,
                                     int* d_block_sums,
                                     int n) {
    // Разделяемая память для одного блока
    // Размер задаётся при запуске ядра
    extern __shared__ int sdata[];

    int tid = threadIdx.x;                      // локальный индекс потока в блоке
    int gid = blockIdx.x * blockDim.x + tid;    // глобальный индекс элемента

    // Загружаем данные из глобальной памяти в shared memory
    // Если индекс выходит за предел массива — используем 0
    int x = (gid < n) ? d_in[gid] : 0;
    sdata[tid] = x;

    // Синхронизация: все потоки блока должны завершить загрузку
    __syncthreads();

    // Upsweep (reduce phase)
    // На этом этапе строится дерево сумм,
    // итоговая сумма блока оказывается в последнем элементе shared memory
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < blockDim.x) {
            sdata[idx] += sdata[idx - offset];
        }
        __syncthreads(); // синхронизация после каждого шага
    }

    // Последний поток сохраняет сумму всего блока
    // и обнуляет последний элемент для downsweep
    if (tid == blockDim.x - 1) {
        d_block_sums[blockIdx.x] = sdata[tid];
        sdata[tid] = 0;
    }
    __syncthreads();

    // Downsweep phase
    // Преобразуем reduce-результат в exclusive scan
    for (int offset = blockDim.x >> 1; offset >= 1; offset >>= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < blockDim.x) {
            int t = sdata[idx - offset];
            sdata[idx - offset] = sdata[idx];
            sdata[idx] += t;
        }
        __syncthreads();
    }

    // Преобразуем exclusive scan в inclusive:
    // прибавляем исходный элемент
    if (gid < n) {
        d_out[gid] = sdata[tid] + x;
    }
}

// GPU-ядро для добавления оффсетов блоков
// Каждый элемент блока получает сумму всех предыдущих блоков
__global__ void add_block_offsets(int* d_out,
                                  const int* d_block_offsets,
                                  int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        d_out[gid] += d_block_offsets[blockIdx.x];
    }
}

int main() {
    const int N = 1'000'000;     // размер массива
    const int BLOCK = 1024;      // размер блока (степень двойки)
    const int GRID = (N + BLOCK - 1) / BLOCK;

    // Генерация входных данных
    std::vector<int> h_in(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 5);

    for (int i = 0; i < N; ++i)
        h_in[i] = dist(rng);

    // CPU-реализация и замер времени
    std::vector<int> h_cpu;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_inclusive_scan(h_in, h_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Выделение памяти на GPU
    int *d_in, *d_out, *d_block_sums, *d_block_offsets;

    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_sums, GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, GRID * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                          N * sizeof(int), cudaMemcpyHostToDevice));

    // Замер времени GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 1) Scan внутри каждого блока
    block_inclusive_scan<<<GRID, BLOCK, BLOCK * sizeof(int)>>>(
        d_in, d_out, d_block_sums, N);

    // 2) Scan сумм блоков на CPU (массив маленький)
    std::vector<int> h_block_sums(GRID), h_block_offsets(GRID);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          GRID * sizeof(int), cudaMemcpyDeviceToHost));

    int acc = 0;
    for (int i = 0; i < GRID; ++i) {
        h_block_offsets[i] = acc;
        acc += h_block_sums[i];
    }

    CUDA_CHECK(cudaMemcpy(d_block_offsets, h_block_offsets.data(),
                          GRID * sizeof(int), cudaMemcpyHostToDevice));

    // 3) Добавление оффсетов
    add_block_offsets<<<GRID, BLOCK>>>(d_out, d_block_offsets, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Проверка корректности
    std::vector<int> h_gpu(N);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out,
                          N * sizeof(int), cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_gpu[i] != h_cpu[i]) {
            correct = false;
            break;
        }
    }

    // Вывод результатов
    printf("Array size: %d\n", N);
    printf("CPU scan time: %.3f ms\n", cpu_ms);
    printf("GPU scan time: %.3f ms\n", gpu_ms);
    printf("Result: %s\n", correct ? "OK" : "FAIL");

    // Освобождение памяти
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    cudaFree(d_block_offsets);

    return 0;
}
