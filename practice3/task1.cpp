#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

// Ядро для слияния отсортированных участков
__global__ void mergeKernel(int* data, int* temp, int width, int n) {
    // Вычисляем глобальный индекс пары, которую будет сливать этот поток
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * width * 2;

    if (start < n) {
        int mid = min(start + width, n);
        int end = min(start + width * 2, n);
        int i = start;
        int j = mid;
        int k = start;

        // Стандартный алгоритм слияния двух упорядоченных списков
        while (i < mid && j < end) {
            if (data[i] <= data[j]) temp[k++] = data[i++];
            else temp[k++] = data[j++];
        }
        while (i < mid) temp[k++] = data[i++];
        while (j < end) temp[k++] = data[j++];
    }
}

void gpuMergeSort(int* h_data, int n) {
    int *d_data, *d_temp;
    size_t size = n * sizeof(int);

    // Выделение памяти на устройстве
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_temp, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Итеративный процесс: ширина подмассивов растет в геометрической прогрессии
    for (int width = 1; width < n; width *= 2) {
        // Вычисляем сколько потоков нужно для текущего шага слияния
        int numThreadsNeeded = (n + (2 * width) - 1) / (2 * width);
        int threadsPerBlock = 256;
        int blocksPerGrid = (numThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

        mergeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_temp, width, n);
        cudaDeviceSynchronize();

        // Меняем указатели местами, чтобы результат стал входными данными для следующего шага
        std::swap(d_data, d_temp);
    }

    // Копируем результат обратно на хост
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_temp);
}

int main() {
    const int n = 1024;
    int data[n];
    for (int i = 0; i < n; i++) data[i] = rand() % 1000;

    gpuMergeSort(data, n);

    std::cout << "Первые 10 элементов: ";
    for (int i = 0; i < 10; i++) std::cout << data[i] << " ";
    return 0;
}