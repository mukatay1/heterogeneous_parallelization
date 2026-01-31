#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
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

// ------------------------------
// Простая пузырьковая сортировка (локальная память потока)
// ------------------------------
template<int K>
__device__ void bubble_sort_local(float (&a)[K]) {
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        #pragma unroll
        for (int j = 0; j < K - 1 - i; ++j) {
            if (a[j] > a[j + 1]) {
                float tmp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = tmp;
            }
        }
    }
}

// ------------------------------
// Kernel 1: локальная сортировка чанков
// Каждый поток берёт K элементов из global memory -> сортирует локально -> пишет назад.
// Это демонстрирует использование "локальной памяти" (регистры/локальные переменные).
// ------------------------------
template<int K>
__global__ void sort_chunks_local(float* d, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * K;
    if (base >= n) return;

    float local[K];

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        int idx = base + i;
        local[i] = (idx < n) ? d[idx] : INFINITY; // padding
    }

    bubble_sort_local<K>(local);

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        int idx = base + i;
        if (idx < n) d[idx] = local[i];
    }
}

// ------------------------------
// Kernel 2: merge двух соседних отсортированных сегментов длины segLen
// Используем shared memory для ускорения чтения и слияния.
// На выходе: d_out содержит объединённые сегменты длиной 2*segLen.
// ------------------------------
__global__ void merge_pass_shared(const float* __restrict__ d_in,
                                 float* __restrict__ d_out,
                                 int n, int segLen) {
    extern __shared__ float sh[];

    int pairId = blockIdx.x; // один блок = одно слияние пары сегментов
    int start = pairId * 2 * segLen;
    if (start >= n) return;

    int mid = min(start + segLen, n);
    int end = min(start + 2 * segLen, n);

    int leftLen  = mid - start;
    int rightLen = end - mid;
    int totalLen = leftLen + rightLen;

    // загрузка двух сегментов в shared
    // sh[0..leftLen-1] - левый, sh[leftLen..leftLen+rightLen-1] - правый
    for (int i = threadIdx.x; i < totalLen; i += blockDim.x) {
        int gidx = start + i;
        sh[i] = d_in[gidx];
    }
    __syncthreads();

    // Параллельный merge здесь сделаем упрощённо:
    // каждый поток пишет несколько элементов, используя "merge path" упрощённый подход.
    // Для учебной работы достаточно: каждый поток делает последовательный merge для своего диапазона.
    // (Это не самый быстрый merge, но демонстрирует shared-memory доступ.)

    int itemsPerThread = (totalLen + blockDim.x - 1) / blockDim.x;
    int outBegin = threadIdx.x * itemsPerThread;
    int outEnd   = min(outBegin + itemsPerThread, totalLen);

    // Для получения корректного merge без сложного merge-path:
    // каждый поток будет вычислять элементы outBegin..outEnd-1 через бинарный поиск разделения.
    // Это чуть сложнее, но работает.

    auto getLeft = [&](int i) -> float { return sh[i]; };
    auto getRight = [&](int i) -> float { return sh[leftLen + i]; };

    auto clamp = [](int x, int lo, int hi) { return max(lo, min(x, hi)); };

    for (int outPos = outBegin; outPos < outEnd; ++outPos) {
        // Хотим найти, сколько взять из left: a, и из right: b, чтобы a+b = outPos
        int aMin = max(0, outPos - rightLen);
        int aMax = min(outPos, leftLen);

        while (aMin < aMax) {
            int a = (aMin + aMax) / 2;
            int b = outPos - a;

            float leftA   = (a > 0) ? getLeft(a - 1) : -INFINITY;
            float leftAn  = (a < leftLen) ? getLeft(a) : INFINITY;
            float rightB  = (b > 0) ? getRight(b - 1) : -INFINITY;
            float rightBn = (b < rightLen) ? getRight(b) : INFINITY;

            if (leftA <= rightBn && rightB <= leftAn) {
                // правильный раздел
                aMin = aMax = a;
            } else if (leftA > rightBn) {
                aMax = a;
            } else {
                aMin = a + 1;
            }
        }

        int a = aMin;
        int b = outPos - a;

        float leftVal  = (a < leftLen) ? getLeft(a) : INFINITY;
        float rightVal = (b < rightLen) ? getRight(b) : INFINITY;

        // Выбираем минимальный из "голов" (для outPos это корректно после разделения)
        float outVal = min(leftVal, rightVal);
        d_out[start + outPos] = outVal;
    }
}

// Проверка (нестрогая): массив должен быть неубывающий
bool is_sorted_non_decreasing(const std::vector<float>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i] < a[i - 1]) return false;
    return true;
}

int main() {
    // Для Task 3 разумно тестировать на меньших данных.
    // Но можно поставить 100000, 1000000 (будет долго из-за пузырька).
    const int N = 100000; // попробуй 10000 / 100000 / 1000000
    const int THREADS = 256;
    const int K = 4; // сколько элементов сортирует один поток локально (пузырьком)

    // host data
    std::vector<float> h(N);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h[i] = dist(rng);

    // device alloc
    float* d_a = nullptr;
    float* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel 1: локальная сортировка чанков K
    int totalThreads = (N + K - 1) / K;
    int blocks = (totalThreads + THREADS - 1) / THREADS;

    cudaEvent_t s1, e1;
    CUDA_CHECK(cudaEventCreate(&s1));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(s1));
    sort_chunks_local<K><<<blocks, THREADS>>>(d_a, N);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(e1));

    float ms_local = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_local, s1, e1));
    std::cout << "Local chunk bubble-sort time: " << ms_local << " ms\n";

    // --- Merge passes: начинаем с сегмента длины K (каждый поток отсортировал K)
    // После merge сегменты увеличиваются в 2 раза: K -> 2K -> 4K -> ...
    int segLen = K;
    bool ping = true; // d_a -> d_b

    while (segLen < N) {
        int pairs = (N + 2 * segLen - 1) / (2 * segLen);
        int mergeBlocks = pairs;
        int mergeThreads = 256; // можно меньше
        size_t shmem = (size_t)min(2 * segLen, N) * sizeof(float); // worst-case для блока

        // В shared нельзя выделять слишком много: ограничение обычно 48/64KB на блок.
        // Поэтому ограничим segLen на merge shared (учебное упрощение).
        if (shmem > 48 * 1024) {
            std::cout << "Stop: shared memory limit reached at segLen=" << segLen
                      << " (shmem=" << shmem / 1024 << " KB). "
                      << "For big N use smaller tiles or multi-stage merge.\n";
            break;
        }

        cudaEvent_t sm, em;
        CUDA_CHECK(cudaEventCreate(&sm));
        CUDA_CHECK(cudaEventCreate(&em));

        CUDA_CHECK(cudaEventRecord(sm));
        if (ping) {
            merge_pass_shared<<<mergeBlocks, mergeThreads, shmem>>>(d_a, d_b, N, segLen);
        } else {
            merge_pass_shared<<<mergeBlocks, mergeThreads, shmem>>>(d_b, d_a, N, segLen);
        }
        CUDA_CHECK(cudaEventRecord(em));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(em));

        float ms_merge = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_merge, sm, em));

        std::cout << "Merge pass segLen=" << segLen << " time: " << ms_merge << " ms\n";

        CUDA_CHECK(cudaEventDestroy(sm));
        CUDA_CHECK(cudaEventDestroy(em));

        ping = !ping;
        segLen *= 2;
    }

    // Copy back result (после цикла результат находится в зависимости от ping/segLen)
    std::vector<float> out(N);
    if (ping) {
        // последняя запись была в d_a
        CUDA_CHECK(cudaMemcpy(out.data(), d_a, N * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(out.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    std::cout << "Sorted check (non-decreasing): "
              << (is_sorted_non_decreasing(out) ? "OK" : "FAIL") << "\n";

    CUDA_CHECK(cudaEventDestroy(s1));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
