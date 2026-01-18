#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Удобный макрос для проверки ошибок CUDA
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// Размер блока (количество потоков в блоке)
#define BLOCK_SIZE 256


// 1. Кернел с использованием ТОЛЬКО ГЛОБАЛЬНОЙ памяти

__global__ void scaleGlobal(float* data, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Чтение из глобальной -> Регистр -> Запись в глобальную
        data[idx] = data[idx] * scalar;
    }
}

// 2. Кернел с использованием РАЗДЕЛЯЕМОЙ (Shared) памяти
__global__ void scaleShared(float* data, float scalar, int n) {
    // Статическое выделение разделяемой памяти на блок
    __shared__ float s_data[BLOCK_SIZE];

    int tid = threadIdx.x; // Локальный индекс внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс

    // 1. Загрузка из глобальной памяти в разделяемую
    if (idx < n) {
        s_data[tid] = data[idx];
    }
    
    // Барьер: ждем, пока все потоки блока загрузят данные
    __syncthreads();

    // 2. Выполнение операции в разделяемой памяти
    // Примечание: проверка idx < n нужна, чтобы лишние потоки не писали мусор,
    // хотя s_data локальна для блока.
    if (idx < n) {
        s_data[tid] *= scalar;
    }

    // Барьер: ждем, пока все вычисления закончатся перед записью
    __syncthreads();

    // 3. Запись результата обратно в глобальную память
    if (idx < n) {
        data[idx] = s_data[tid];
    }
}

int main() {
    int n = 1000000; // 1 миллион элементов
    size_t bytes = n * sizeof(float);
    float scalar = 2.0f;

    // Выделение памяти на хосте
    std::vector<float> h_data(n);
    for (int i = 0; i < n; i++) {
        h_data[i] = 1.0f; // Заполняем единицами
    }

    // Выделение памяти на устройстве (GPU)
    float *d_data_global, *d_data_shared;
    CHECK_CUDA(cudaMalloc(&d_data_global, bytes));
    CHECK_CUDA(cudaMalloc(&d_data_shared, bytes));

    // Копирование данных Host -> Device
    CHECK_CUDA(cudaMemcpy(d_data_global, h_data.data(), bytes, cudaMemcpyHostToDevice));
    // Копируем те же данные во второй буфер для честного сравнения
    CHECK_CUDA(cudaMemcpy(d_data_shared, h_data.data(), bytes, cudaMemcpyHostToDevice));

    // Настройка сетки запуска
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Создание событий для замера времени
   cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Обработка массива из " << n << " элементов.\n" << "------------------------------------------------\n";

    //  ЗАПУСК ВЕРСИИ 1: GLOBAL MEMORY 
    cudaEventRecord(start);
    scaleGlobal<<<gridSize, blockSize>>>(d_data_global, scalar, n);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float millisecondsGlobal = 0;
    cudaEventElapsedTime(&millisecondsGlobal, start, stop);
    
    std::cout << "1. Global Memory Kernel: " << millisecondsGlobal << " ms" << std::endl;

    // ЗАПУСК ВЕРСИИ 2: SHARED MEMORY
    cudaEventRecord(start);
    scaleShared<<<gridSize, blockSize>>>(d_data_shared, scalar, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float millisecondsShared = 0;
    cudaEventElapsedTime(&millisecondsShared, start, stop);

    std::cout << "2. Shared Memory Kernel: " << millisecondsShared << " ms" << std::endl;

    // Проверка результатов (опционально, копируем обратно первый элемент)
    float h_result;
    cudaMemcpy(&h_result, d_data_shared, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Проверка результата (1.0 * 2.0): " << h_result << std::endl;

    // Освобождение памяти
    cudaFree(d_data_global);
    cudaFree(d_data_shared);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}