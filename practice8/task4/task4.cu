#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>

#define N 1000000   // Можно менять размер для экспериментов

// ---------------- CUDA ядро ----------------
// Умножение каждого элемента на 2
__global__ void gpu_mul2(float* data, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] *= 2.0f;
}

// ---------------- CPU функция (OpenMP) ----------------
void cpu_mul2(float* data, int size)
{
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
        data[i] *= 2.0f;
}

// ---------------- Проверка корректности ----------------
bool check_all_2(const std::vector<float>& a)
{
    // Проверяем несколько точек (быстро), что примерно равно 2.0
    const float eps = 1e-5f;
    int idxs[] = {0, (int)a.size()/2, (int)a.size()-1};
    for (int k = 0; k < 3; k++)
    {
        if (std::fabs(a[idxs[k]] - 2.0f) > eps) return false;
    }
    return true;
}

int main()
{
    std::cout << "Размер массива N = " << N << "\n";

    // Чтобы измерения были честнее, сделаем несколько прогонов и возьмем минимум
    const int RUNS = 5;

    double best_cpu = 1e9;
    double best_gpu = 1e9;
    double best_hybrid = 1e9;

    for (int r = 0; r < RUNS; r++)
    {
        // ===================== 1) CPU (OpenMP) =====================
        {
            std::vector<float> h(N, 1.0f);

            double t0 = omp_get_wtime();
            cpu_mul2(h.data(), N);
            double t1 = omp_get_wtime();

            if (!check_all_2(h))
                std::cerr << "[CPU] Ошибка результата!\n";

            best_cpu = std::min(best_cpu, (t1 - t0));
        }

        // ===================== 2) GPU (CUDA) =====================
        {
            std::vector<float> h(N, 1.0f);

            float* d = nullptr;
            cudaMalloc((void**)&d, N * sizeof(float));

            // ВАЖНО: сюда включаем передачу данных, потому что в задании есть копирование туда-обратно
            double t0 = omp_get_wtime();

            cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks  = (N + threads - 1) / threads;
            gpu_mul2<<<blocks, threads>>>(d, N);
            cudaDeviceSynchronize();

            cudaMemcpy(h.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost);

            double t1 = omp_get_wtime();

            if (!check_all_2(h))
                std::cerr << "[GPU] Ошибка результата!\n";

            cudaFree(d);

            best_gpu = std::min(best_gpu, (t1 - t0));
        }

        // ===================== 3) HYBRID (CPU + GPU) =====================
        {
            std::vector<float> h(N, 1.0f);

            int cpu_size = N / 2;
            int gpu_size = N - cpu_size;

            float* d = nullptr;
            cudaMalloc((void**)&d, gpu_size * sizeof(float));

            // Копируем вторую половину на GPU (это часть накладных расходов гибрида)
            cudaMemcpy(d, h.data() + cpu_size,
                       gpu_size * sizeof(float),
                       cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks  = (gpu_size + threads - 1) / threads;

            double t0 = omp_get_wtime();

            // CPU считает первую половину
            cpu_mul2(h.data(), cpu_size);

            // GPU считает вторую половину
            gpu_mul2<<<blocks, threads>>>(d, gpu_size);
            cudaDeviceSynchronize();

            // Возвращаем вторую половину обратно
            cudaMemcpy(h.data() + cpu_size, d,
                       gpu_size * sizeof(float),
                       cudaMemcpyDeviceToHost);

            double t1 = omp_get_wtime();

            if (!check_all_2(h))
                std::cerr << "[HYBRID] Ошибка результата!\n";

            cudaFree(d);

            best_hybrid = std::min(best_hybrid, (t1 - t0));
        }
    }

    // ---------------- Вывод результатов ----------------
    std::cout << "\nЛучшее время из " << RUNS << " прогонов:\n";
    std::cout << "CPU (OpenMP):   " << best_cpu    << " сек\n";
    std::cout << "GPU (CUDA):     " << best_gpu    << " сек (с копированием H<->D)\n";
    std::cout << "HYBRID (1/2+1/2): " << best_hybrid << " сек\n";

    // ---------------- Ускорения ----------------
    // Аккуратно: делим по цифрам не надо, тут double; покажем коэффициенты как есть
    std::cout << "\nУскорение GPU относительно CPU:     " << (best_cpu / best_gpu) << "x\n";
    std::cout << "Ускорение HYBRID относительно CPU:  " << (best_cpu / best_hybrid) << "x\n";

    // ---------------- Простейший анализ (текстом) ----------------
    std::cout << "\nКороткий анализ:\n";
    std::cout << "- Если N маленький, GPU/Hybrid могут быть медленнее из-за копирования по PCIe.\n";
    std::cout << "- При большом N GPU обычно выигрывает, потому что вычисления параллельные.\n";
    std::cout << "- Hybrid полезен, когда:\n";
    std::cout << "  (1) есть достаточно работы и для CPU, и для GPU,\n";
    std::cout << "  (2) копирований минимум,\n";
    std::cout << "  (3) CPU и GPU реально заняты одновременно.\n";

    return 0;
}
