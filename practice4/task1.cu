#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

// Макрос проверки ошибок
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cerr << "CRASH at line " << __LINE__ << ": " \
                  << cudaGetErrorString(error) << " (Code: " << (int)error << ")" << std::endl; \
        exit(1); \
    } \
}

// Ядро сортировки
__global__ void mergeKernel(int* data, int* temp, long width, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long start = idx * width * 2;

    if (start < n) {
        long mid = min(start + width, n);
        long end = min(start + width * 2, n);
        long i = start, j = mid, k = start;

        while (i < mid && j < end) {
            if (data[i] <= data[j]) temp[k++] = data[i++];
            else temp[k++] = data[j++];
        }
        while (i < mid) temp[k++] = data[i++];
        while (j < end) temp[k++] = data[j++];
    }
}

void run_test(int n) {
    size_t bytes = n * sizeof(int);
    std::vector<int> h_data(n);
    for(int i=0; i<n; i++) h_data[i] = rand() % 10000;

    int *d_data, *d_temp;
    
    // ВЫДЕЛЕНИЕ ПАМЯТИ (Здесь у тебя падало)
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMalloc(&d_temp, bytes));
    
    CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (long width = 1; width < n; width *= 2) {
        long numMerges = (n + width * 2 - 1) / (width * 2);
        int threads = 256;
        int blocks = (numMerges + threads - 1) / threads;

        mergeKernel<<<blocks, threads>>>(d_data, d_temp, width, n);
        CHECK(cudaGetLastError());
        std::swap(d_data, d_temp);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    bool correct = std::is_sorted(h_data.begin(), h_data.end());
    
    printf("Size: %d | Time: %.4f ms | Status: %s\n", n, ms, correct ? "OK" : "FAIL");

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_temp));
}

int main() {
    // === БЛОК ПРИНУДИТЕЛЬНОГО ПОИСКА NVIDIA ===
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "CRITICAL: No CUDA devices found! Driver problem?" << std::endl;
        return 1;
    }

    int selectedDevice = -1;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        
        // Ищем карту, похожую на GTX
        std::string name = prop.name;
        if (name.find("GeForce") != std::string::npos || name.find("NVIDIA") != std::string::npos) {
            selectedDevice = i;
        }
    }

    if (selectedDevice != -1) {
        cudaSetDevice(selectedDevice);
        std::cout << ">>> FORCING USAGE OF DEVICE: " << selectedDevice << std::endl;
    } else {
        std::cerr << "WARNING: No NVIDIA GeForce found. Trying default..." << std::endl;
    }
    // ==========================================

    run_test(10000);
    run_test(100000);
    
    return 0;
}