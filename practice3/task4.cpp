#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>

// CUDA Ядро для слияния 
__global__ void gpu_merge_step(int* data, int* temp, int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * width * 2;
    if (start < n) {
        int mid = min(start + width, n);
        int end = min(start + width * 2, n);
        int i = start, j = mid, k = start;
        while (i < mid && j < end) {
            temp[k++] = (data[i] <= data[j]) ? data[i++] : data[j++];
        }
        while (i < mid) temp[k++] = data[i++];
        while (j < end) temp[k++] = data[j++];
    }
}

int main() {
    const int n = 1000000; // Размер массива
    size_t bytes = n * sizeof(int);

    // Подготовка данных
    std::vector<int> h_data(n), h_result_cpu(n);
    for (int i = 0; i < n; i++) h_data[i] = rand() % 100000;
    h_result_cpu = h_data;

    // 1. Тест CPU (Последовательно)
    clock_t start_cpu = clock();
    std::sort(h_result_cpu.begin(), h_result_cpu.end());
    clock_t end_cpu = clock();
    double time_cpu = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    // 2. Тест GPU (CUDA)
    int *d_data, *d_temp;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_temp, bytes);
    
    float time_gpu;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);

    for (int width = 1; width < n; width *= 2) {
        int threadsPerBlock = 256;
        int blocks = (n / (2 * width) + 1);
        gpu_merge_step<<<blocks, threadsPerBlock>>>(d_data, d_temp, width, n);
        std::swap(d_data, d_temp);
    }

    cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu, start, stop);

    // Вывод результатов
    std::cout << "Массив: " << n << " элементов" << std::endl;
    std::cout << "Время CPU: " << time_cpu << " сек." << std::endl;
    std::cout << "Время GPU: " << time_gpu / 1000.0 << " сек. (с учетом копирования)" << std::endl;
    std::cout << "Ускорение: " << time_cpu / (time_gpu / 1000.0) << "x" << std::endl;

    cudaFree(d_data); cudaFree(d_temp);
    return 0;
}