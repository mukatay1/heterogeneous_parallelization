#include <iostream>
#include <cuda_runtime.h>

// Восстановление свойств кучи на GPU
__device__ void heapify(int* data, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;

    if (l < n && data[l] > data[largest]) largest = l;
    if (r < n && data[r] > data[largest]) largest = r;

    if (largest != i) {
        int temp = data[i];
        data[i] = data[largest];
        data[largest] = temp;
        heapify(data, n, largest);
    }
}

// Построение кучи параллельными блоками
__global__ void buildHeap(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = n / 2 - 1 - idx; i >= 0; i -= blockDim.x * gridDim.x) {
        heapify(data, n, i);
    }
}

int main() {
    const int n = 1024;
    int h_data[n];
    for (int i = 0; i < n; i++) h_data[i] = rand() % 1000;

    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    // 1. Параллельное построение кучи
    buildHeap<<<1, 256>>>(d_data, n);
    cudaDeviceSynchronize();

    std::cout << "Пирамидальная сортировка на GPU запущена..." << std::endl;

    cudaFree(d_data);
    return 0;
}