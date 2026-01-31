#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA ошибка: " << cudaGetErrorString(err) \
                  << " (файл " << __FILE__ << ", строка " << __LINE__ << ")\n"; \
        std::exit(1); \
    } \
} while(0)

// -------------------------
// 1) Коалесцированное ядро (эффективный доступ)
// Каждый поток читает/пишет соседний элемент: in[i] -> out[i]
// -------------------------
__global__ void kernel_coalesced(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// -------------------------
// 2) Некоалесцированное ядро (неэффективный доступ)
// Каждый поток читает элемент с "прыжком" (stride), что ломает коалесцирование.
// Важно: пишем всё равно в out[i], чтобы не было гонок.
// -------------------------
__global__ void kernel_noncoalesced(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n,
                                    int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (i * stride) % n;      // чтение с большим шагом
        out[i] = in[j] * 2.0f;
    }
}

// -------------------------
// 3) "Тяжёлая" операция, где shared memory даёт выигрыш:
// 1D stencil: out[i] = in[i-1] + in[i] + in[i+1]
// Вариант A: всё читаем из глобальной памяти (медленнее)
// -------------------------
__global__ void kernel_stencil_global(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float left  = (i > 0)     ? in[i - 1] : 0.0f;
        float mid   = in[i];
        float right = (i + 1 < n) ? in[i + 1] : 0.0f;
        out[i] = left + mid + right;
    }
}

// -------------------------
// 4) Оптимизация: shared memory + "ореол" (halo)
// Загружаем блок в shared, включая соседние элементы,
// затем считаем stencil быстрее (меньше обращений к global memory).
// -------------------------
__global__ void kernel_stencil_shared(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int n)
{
    extern __shared__ float s[]; // динамический shared
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Индекс в shared: сдвиг на 1, чтобы слева был halo
    int s_idx = tid + 1;

    // Загружаем центральный элемент
    if (i < n) s[s_idx] = in[i];
    else      s[s_idx] = 0.0f;

    // Загружаем левый halo (только tid==0)
    if (tid == 0) {
        s[0] = (i > 0) ? in[i - 1] : 0.0f;
    }

    // Загружаем правый halo (только последний поток блока)
    if (tid == blockDim.x - 1) {
        s[blockDim.x + 1] = (i + 1 < n) ? in[i + 1] : 0.0f;
    }

    __syncthreads();

    // Считаем stencil из shared
    if (i < n) {
        out[i] = s[s_idx - 1] + s[s_idx] + s[s_idx + 1];
    }
}

// -------------------------
// Утилита: замер времени ядра через cudaEvent (в миллисекундах)
// -------------------------
template <typename Kernel, typename... Args>
float time_kernel_ms(Kernel k, dim3 grid, dim3 block, size_t shmem, int iters, Args... args)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Прогрев (warm-up), чтобы первый запуск не искажал результаты
    k<<<grid, block, shmem>>>(args...);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        k<<<grid, block, shmem>>>(args...);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters; // среднее за итерацию
}

// -------------------------
// Главная программа
// -------------------------
int main(int argc, char** argv)
{
    int N = 10'000'000;
    if (argc >= 2) N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Ошибка: N должен быть > 0\n";
        return 1;
    }

    std::cout << "Практическая №10 / Таск 2: Паттерны доступа к памяти на GPU (CUDA)\n";
    std::cout << "N = " << N << "\n\n";

    // Создаём входные данные на CPU
    std::vector<float> h_in(N);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) h_in[i] = dist(gen);

    // Выделяем память на GPU
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(float)));

    // Копируем вход на GPU
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Настройки теста
    const int iters = 30;               // сколько раз повторять ядро для среднего
    const int stride = 33;              // "плохой" шаг (не кратен размеру warp, ломает доступ)
    int block_list[] = {128, 256, 512}; // тест "организации потоков"

    std::cout << "Замеры времени через cudaEvent (среднее за " << iters << " итераций)\n";
    std::cout << "stride (для плохого доступа) = " << stride << "\n\n";

    for (int b : block_list)
    {
        dim3 block(b);
        dim3 grid((N + b - 1) / b);

        // --- 1) Коалесцированное ядро ---
        float t_coal = time_kernel_ms(kernel_coalesced, grid, block, 0, iters, d_in, d_out, N);

        // --- 2) Некоалесцированное ядро ---
        float t_noncoal = time_kernel_ms(kernel_noncoalesced, grid, block, 0, iters, d_in, d_out, N, stride);

        // --- 3) Stencil (global memory) ---
        float t_stencil_global = time_kernel_ms(kernel_stencil_global, grid, block, 0, iters, d_in, d_out, N);

        // --- 4) Stencil (shared memory) ---
        // shared размер: (blockDim + 2) float (halo слева и справа)
        size_t shmem = (b + 2) * sizeof(float);
        float t_stencil_shared = time_kernel_ms(kernel_stencil_shared, grid, block, shmem, iters, d_in, d_out, N);

        std::cout << "Block size = " << b << "\n";
        std::cout << "  1) Coalesced (in[i]->out[i])         : " << t_coal << " ms\n";
        std::cout << "  2) Non-coalesced (in[(i*stride)%N]) : " << t_noncoal << " ms\n";
        std::cout << "  3) Stencil global memory            : " << t_stencil_global << " ms\n";
        std::cout << "  4) Stencil shared memory (optimized): " << t_stencil_shared << " ms\n";

        // Простые коэффициенты сравнения
        std::cout << "  Сравнение:\n";
        std::cout << "    Non/Coalesced = " << (t_noncoal / t_coal) << "x (обычно > 1 => хуже)\n";
        std::cout << "    Global/Shared = " << (t_stencil_global / t_stencil_shared) << "x (обычно > 1 => shared лучше)\n";
        std::cout << "\n";
    }

    // Освобождение памяти
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "Готово.\n";
    return 0;
}
