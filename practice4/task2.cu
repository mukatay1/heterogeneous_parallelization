#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Локальная быстрая сортировка для одного потока
__device__ void localQuicksort(int* data, int left, int right) {
    if (left >= right) return;
    
    int pivot = data[(left + right) / 2];
    int i = left, j = right;
    
    while (i <= j) {
        while (data[i] < pivot) i++;
        while (data[j] > pivot) j--;
        if (i <= j) {
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
            j--;
        }
    }
    if (left < j) localQuicksort(data, left, j);
    if (i < right) localQuicksort(data, i, right);
}

// Ядро, распределяющее части массива по потокам
__global__ void quicksortKernel(int* data, int n, int chunkSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = idx * chunkSize;
    int right = left + chunkSize - 1;

    if (left < n) {
        if (right >= n) right = n - 1;
        localQuicksort(data, left, right);
    }
}

int main() {
    const int n = 10000;
    const int chunkSize = 500; // Каждый поток сортирует 500 элементов
    size_t size = n * sizeof(int);
    int h_data[n];

    for (int i = 0; i < n; i++) h_data[i] = rand() % 10000;

    int* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n / chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    // Запуск параллельной сортировки кусков
    quicksortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, chunkSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    // В конце на CPU выполняется финальное слияние или досортировка
    std::cout << "Параллельная сортировка частей завершена." << std::endl;

    cudaFree(d_data);
    return 0;
}