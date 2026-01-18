#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Удобный макрос для проверки ошибок CUDA
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// Размер массива (1 миллион элементов)
#define N 1000000


// Кернел для поэлементного сложения массивов
// Каждый поток обрабатывает один элемент:
// C[i] = A[i] + B[i]
__global__ void addArrays(const float* a, const float* b, float* c, int n) {

    // Глобальный индекс элемента
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {

        // Чтение из глобальной памяти -> регистры -> запись в глобальную память
        c[idx] = a[idx] + b[idx];
    }
}

int main() {

    size_t bytes = N * sizeof(float);

    // Выделение памяти на хосте (CPU)
    std::vector<float> h_a(N), h_b(N), h_c(N);

    // Инициализация входных массивов
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Выделение памяти на устройстве (GPU)
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Копирование данных Host -> Device
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << "Поэлементное сложение массивов из " << N << " элементов\n";
    std::cout << "------------------------------------------------------\n";

    // Набор размеров блока потоков для исследования производительности
    int blockSizes[] = {128, 256, 512};

    // Создание CUDA-событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Перебор различных размеров блока
    for (int blockSize : blockSizes) {

        // Настройка конфигурации запуска
        dim3 block(blockSize);
        dim3 grid((N + blockSize - 1) / blockSize);

        // Запуск кернела и замер времени
        cudaEventRecord(start);
        addArrays<<<grid, block>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);

        // Ожидание завершения вычислений
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "Block size " << blockSize
                  << ": " << milliseconds << " ms" << std::endl;
    }

    std::cout << "------------------------------------------------------\n";

    // Копирование результата обратно на хост
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Проверка корректности результата (первый элемент)
    std::cout << "Проверка результата (1.0 + 2.0): "
              << h_c[0] << std::endl;

    // Освобождение памяти
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
