// reduction_kernel.cu
#include <cuda_runtime.h>

// Ядро редукции: каждая блок считает частичную сумму
__global__ void reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float sdata[]; // разделяемая память

    // Глобальный индекс
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Загружаем данные в shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Параллельная редукция внутри блока
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Поток 0 записывает результат блока
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
