#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// Ядро для слияния подмассивов
__global__ void mergeShared(int* data, int* temp, int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * width * 2;

    if (start < n) {
        int mid = min(start + width, n);
        int end = min(start + 2 * width, n);
        
        int i = start;
        int j = mid;
        
        for (int k = start; k < end; k++) {
            if (i < mid && (j >= end || data[i] <= data[j])) {
                temp[k] = data[i];
                i++;
            } else {
                temp[k] = data[j];
                j++;
            }
        }
    }
}

void cudaMergeSort(int* h_data, int n) {
    int *d_data, *d_temp;
    size_t size = n * sizeof(int);

    cudaMalloc(&d_data, size);
    cudaMalloc(&d_temp, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Итеративное слияние: 1, 2, 4, 8...
    for (int width = 1; width < n; width *= 2) {
        int numThreads = (n + (2 * width) - 1) / (2 * width);
        int threadsPerBlock = 256;
        int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

        mergeShared<<<blocks, threadsPerBlock>>>(d_data, d_temp, width, n);
        
        // Меняем указатели местами, чтобы избежать лишнего копирования
        int* temp = d_data;
        d_data = d_temp;
        d_temp = temp;
    }

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_temp);
}

void runCudaTest(int n) {
    vector<int> data(n);
    for(int i = 0; i < n; i++) data[i] = rand() % 1000;

    auto start = chrono::high_resolution_clock::now();
    cudaMergeSort(data.data(), n);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsed = end - start;
    cout << "Размер: " << n << " | Время GPU: " << elapsed.count() << " ms" << endl;
}

int main() {
    runCudaTest(10000);
    runCudaTest(100000);
    return 0;
}