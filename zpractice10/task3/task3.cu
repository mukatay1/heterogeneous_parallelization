#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA ошибка: " << cudaGetErrorString(err) \
                  << " (файл " << __FILE__ << ", строка " << __LINE__ << ")\n"; \
        std::exit(1); \
    } \
} while(0)

// ---------------- CUDA ядро ----------------
// Простейшая обработка: умножение на 2
__global__ void mul2_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// ---------------- CPU обработка (OpenMP) ----------------
static void cpu_mul2(float* data, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        data[i] *= 2.0f;
}

// ---------------- Заполнение массива ----------------
static void fill_random(std::vector<float>& a)
{
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &x : a) x = dist(gen);
}

// ---------------- Утилита для замера cudaEvent (мс) ----------------
static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ---------------- БАЗОВЫЙ ВАРИАНТ (без оптимизаций) ----------------
// Синхронные копирования + один stream (по сути, всё последовательно)
static void run_baseline_sync(float* h_all, int N, int cpu_size, int gpu_size)
{
    std::cout << "\n=== Baseline: sync memcpy + kernel (без overlap) ===\n";

    // GPU часть — это вторая половина массива
    float* h_gpu = h_all + cpu_size;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  gpu_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, gpu_size * sizeof(float)));

    cudaEvent_t e_h2d_s, e_h2d_e, e_k_s, e_k_e, e_d2h_s, e_d2h_e;
    CUDA_CHECK(cudaEventCreate(&e_h2d_s));
    CUDA_CHECK(cudaEventCreate(&e_h2d_e));
    CUDA_CHECK(cudaEventCreate(&e_k_s));
    CUDA_CHECK(cudaEventCreate(&e_k_e));
    CUDA_CHECK(cudaEventCreate(&e_d2h_s));
    CUDA_CHECK(cudaEventCreate(&e_d2h_e));

    double t0_total = omp_get_wtime();

    // CPU часть считаем (первая половина)
    double t0_cpu = omp_get_wtime();
    cpu_mul2(h_all, cpu_size);
    double t1_cpu = omp_get_wtime();

    // GPU часть: H2D -> kernel -> D2H (последовательно)
    CUDA_CHECK(cudaEventRecord(e_h2d_s));
    CUDA_CHECK(cudaMemcpy(d_in, h_gpu, gpu_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e_h2d_e));
    CUDA_CHECK(cudaEventSynchronize(e_h2d_e));

    int threads = 256;
    int blocks  = (gpu_size + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(e_k_s));
    mul2_kernel<<<blocks, threads>>>(d_in, d_out, gpu_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e_k_e));
    CUDA_CHECK(cudaEventSynchronize(e_k_e));

    CUDA_CHECK(cudaEventRecord(e_d2h_s));
    CUDA_CHECK(cudaMemcpy(h_gpu, d_out, gpu_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e_d2h_e));
    CUDA_CHECK(cudaEventSynchronize(e_d2h_e));

    double t1_total = omp_get_wtime();

    float t_h2d = elapsed_ms(e_h2d_s, e_h2d_e);
    float t_k   = elapsed_ms(e_k_s, e_k_e);
    float t_d2h = elapsed_ms(e_d2h_s, e_d2h_e);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU time:    " << (t1_cpu - t0_cpu) * 1000.0 << " ms\n";
    std::cout << "H2D time:    " << t_h2d << " ms\n";
    std::cout << "Kernel time: " << t_k   << " ms\n";
    std::cout << "D2H time:    " << t_d2h << " ms\n";
    std::cout << "Total time:  " << (t1_total - t0_total) * 1000.0 << " ms\n";

    std::cout << "Overhead (H2D+D2H): " << (t_h2d + t_d2h) << " ms\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    CUDA_CHECK(cudaEventDestroy(e_h2d_s));
    CUDA_CHECK(cudaEventDestroy(e_h2d_e));
    CUDA_CHECK(cudaEventDestroy(e_k_s));
    CUDA_CHECK(cudaEventDestroy(e_k_e));
    CUDA_CHECK(cudaEventDestroy(e_d2h_s));
    CUDA_CHECK(cudaEventDestroy(e_d2h_e));
}

// ---------------- ОПТИМИЗИРОВАННЫЙ ВАРИАНТ ----------------
// 1) Используем pinned memory (cudaHostAlloc) -> быстрее копирование
// 2) Используем cudaMemcpyAsync + 2 streams (pipeline по чанкам)
// 3) CPU считает свою часть параллельно с GPU работой (overlap)
static void run_optimized_async_pipeline(const std::vector<float>& src,
                                         float* h_pinned, // pinned буфер для всего массива
                                         int N, int cpu_size, int gpu_size,
                                         int chunk_elems)
{
    std::cout << "\n=== Optimized: pinned + cudaMemcpyAsync + 2 streams (pipeline) ===\n";

    // Копируем исходные данные в pinned память (один раз)
    // (в реальном приложении данные могут уже быть в pinned)
    std::copy(src.begin(), src.end(), h_pinned);

    float* h_all = h_pinned;
    float* h_gpu = h_all + cpu_size; // GPU часть — вторая половина

    // На GPU выделим 2 набора буферов (double buffering) под чанки
    float *d_in[2]  = {nullptr, nullptr};
    float *d_out[2] = {nullptr, nullptr};

    CUDA_CHECK(cudaMalloc((void**)&d_in[0],  chunk_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out[0], chunk_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_in[1],  chunk_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out[1], chunk_elems * sizeof(float)));

    cudaStream_t s[2];
    CUDA_CHECK(cudaStreamCreate(&s[0]));
    CUDA_CHECK(cudaStreamCreate(&s[1]));

    // События для грубой оценки накладных расходов (на весь GPU-пайплайн)
    cudaEvent_t e_total_s, e_total_e;
    CUDA_CHECK(cudaEventCreate(&e_total_s));
    CUDA_CHECK(cudaEventCreate(&e_total_e));

    // Дополнительно посчитаем суммарное время копирований и ядер по чанкам (как метрики)
    float total_h2d_ms = 0.0f;
    float total_d2h_ms = 0.0f;
    float total_k_ms   = 0.0f;

    double t0_total = omp_get_wtime();

    // Запускаем GPU пайплайн
    CUDA_CHECK(cudaEventRecord(e_total_s, 0));

    // Параллельно с GPU будем считать CPU часть
    double t0_cpu = omp_get_wtime();

    // CPU обработка первой половины
    cpu_mul2(h_all, cpu_size);

    double t1_cpu = omp_get_wtime();

    // GPU: обрабатываем вторую половину чанками
    int remaining = gpu_size;
    int offset = 0;

    // Для замеров по чанкам используем events (создадим один комплект и переиспользуем)
    cudaEvent_t e_h2d_s, e_h2d_e, e_k_s, e_k_e, e_d2h_s, e_d2h_e;
    CUDA_CHECK(cudaEventCreate(&e_h2d_s));
    CUDA_CHECK(cudaEventCreate(&e_h2d_e));
    CUDA_CHECK(cudaEventCreate(&e_k_s));
    CUDA_CHECK(cudaEventCreate(&e_k_e));
    CUDA_CHECK(cudaEventCreate(&e_d2h_s));
    CUDA_CHECK(cudaEventCreate(&e_d2h_e));

    int chunk_id = 0;
    while (remaining > 0)
    {
        int cur = std::min(chunk_elems, remaining);
        int buf = chunk_id & 1; // 0 или 1 — чередуем буферы и стримы

        // Асинхронное копирование H2D
        CUDA_CHECK(cudaEventRecord(e_h2d_s, s[buf]));
        CUDA_CHECK(cudaMemcpyAsync(d_in[buf],
                                   h_gpu + offset,
                                   cur * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   s[buf]));
        CUDA_CHECK(cudaEventRecord(e_h2d_e, s[buf]));

        // Ядро
        int threads = 256;
        int blocks  = (cur + threads - 1) / threads;

        CUDA_CHECK(cudaEventRecord(e_k_s, s[buf]));
        mul2_kernel<<<blocks, threads, 0, s[buf]>>>(d_in[buf], d_out[buf], cur);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e_k_e, s[buf]));

        // Асинхронное копирование D2H
        CUDA_CHECK(cudaEventRecord(e_d2h_s, s[buf]));
        CUDA_CHECK(cudaMemcpyAsync(h_gpu + offset,
                                   d_out[buf],
                                   cur * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   s[buf]));
        CUDA_CHECK(cudaEventRecord(e_d2h_e, s[buf]));

        // Дадим этому стриму закончить текущий чанк, чтобы корректно снять времена
        // (В реальной жизни можно не синхронизировать каждый чанк, но для профилирования — удобно)
        CUDA_CHECK(cudaEventSynchronize(e_d2h_e));

        total_h2d_ms += elapsed_ms(e_h2d_s, e_h2d_e);
        total_k_ms   += elapsed_ms(e_k_s,   e_k_e);
        total_d2h_ms += elapsed_ms(e_d2h_s, e_d2h_e);

        offset += cur;
        remaining -= cur;
        chunk_id++;
    }

    // Дожидаемся завершения всех стримов
    CUDA_CHECK(cudaStreamSynchronize(s[0]));
    CUDA_CHECK(cudaStreamSynchronize(s[1]));

    CUDA_CHECK(cudaEventRecord(e_total_e, 0));
    CUDA_CHECK(cudaEventSynchronize(e_total_e));

    double t1_total = omp_get_wtime();
    float gpu_total_ms = elapsed_ms(e_total_s, e_total_e);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU time (параллельно с GPU): " << (t1_cpu - t0_cpu) * 1000.0 << " ms\n";
    std::cout << "GPU pipeline total (events):  " << gpu_total_ms << " ms\n";
    std::cout << "GPU sums (по чанкам):\n";
    std::cout << "  H2D sum:    " << total_h2d_ms << " ms\n";
    std::cout << "  Kernel sum: " << total_k_ms   << " ms\n";
    std::cout << "  D2H sum:    " << total_d2h_ms << " ms\n";
    std::cout << "Overhead (H2D+D2H) sum: " << (total_h2d_ms + total_d2h_ms) << " ms\n";

    std::cout << "Total wall time (omp_get_wtime): " << (t1_total - t0_total) * 1000.0 << " ms\n";

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(e_total_s));
    CUDA_CHECK(cudaEventDestroy(e_total_e));

    CUDA_CHECK(cudaEventDestroy(e_h2d_s));
    CUDA_CHECK(cudaEventDestroy(e_h2d_e));
    CUDA_CHECK(cudaEventDestroy(e_k_s));
    CUDA_CHECK(cudaEventDestroy(e_k_e));
    CUDA_CHECK(cudaEventDestroy(e_d2h_s));
    CUDA_CHECK(cudaEventDestroy(e_d2h_e));

    CUDA_CHECK(cudaStreamDestroy(s[0]));
    CUDA_CHECK(cudaStreamDestroy(s[1]));

    CUDA_CHECK(cudaFree(d_in[0]));
    CUDA_CHECK(cudaFree(d_out[0]));
    CUDA_CHECK(cudaFree(d_in[1]));
    CUDA_CHECK(cudaFree(d_out[1]));
}

int main(int argc, char** argv)
{
    int N = 20'000'000;
    if (argc >= 2) N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Ошибка: N должен быть > 0\n";
        return 1;
    }

    // Делим работу: половина CPU, половина GPU
    int cpu_size = N / 2;
    int gpu_size = N - cpu_size;

    std::cout << "Практическая №10 / Таск 3: Гибрид CPU+GPU (профилирование)\n";
    std::cout << "N = " << N << " (CPU: " << cpu_size << ", GPU: " << gpu_size << ")\n";
    std::cout << "Требования: hybrid + cudaMemcpyAsync + streams + профилирование + оптимизация\n\n"; // :contentReference[oaicite:4]{index=4}

    // Исходные данные (обычная память)
    std::vector<float> h(N);
    fill_random(h);

    // ---------------- 1) Baseline ----------------
    // Для baseline используем обычный вектор (как есть)
    // ВНИМАНИЕ: baseline изменяет данные (умножает на 2),
    // поэтому перед каждым запуском лучше иметь копию исходного массива.
    std::vector<float> h1 = h;
    run_baseline_sync(h1.data(), N, cpu_size, gpu_size);

    // ---------------- 2) Optimized ----------------
    // Оптимизация: pinned memory + async + streams + chunking
    float* h_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&h_pinned, N * sizeof(float), cudaHostAllocDefault));

    // Размер чанка (можно менять для экспериментов)
    int chunk_elems = 1 << 20; // ~1M элементов = 4MB (нормально для PCIe)
    std::vector<float> h2 = h;
    run_optimized_async_pipeline(h2, h_pinned, N, cpu_size, gpu_size, chunk_elems);

    CUDA_CHECK(cudaFreeHost(h_pinned));

    std::cout << "\nГотово.\n";
    return 0;
}
