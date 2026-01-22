// task5_stack_queue.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s (%d): %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        std::exit(1);                                                   \
    }                                                                   \
} while (0)

// =====================================================
//                   ПАРАЛЛЕЛЬНЫЙ СТЕК (GPU)
// =====================================================
struct Stack {
    int* data;       // массив стека в глобальной памяти GPU
    int  top;        // -1 => пусто, иначе индекс верхнего элемента
    int  capacity;   // емкость

    __device__ void init(int* buffer, int size) {
        data = buffer;
        top = -1;
        capacity = size;
    }

    __device__ bool push(int value) {
        int old = atomicAdd(&top, 1);
        int pos = old + 1;

        if (pos < capacity) {
            data[pos] = value;
            return true;
        } else {
            atomicSub(&top, 1); // откат
            return false;
        }
    }

    __device__ bool pop(int* value) {
        int old = atomicSub(&top, 1);
        int pos = old;

        if (pos >= 0) {
            *value = data[pos];
            return true;
        } else {
            atomicAdd(&top, 1); // откат
            return false;
        }
    }
};

__global__ void stack_init_kernel(Stack* s, int* buf, int cap) {
    if (blockIdx.x == 0 && threadIdx.x == 0) s->init(buf, cap);
}

__global__ void stack_push_kernel(Stack* s, int n_ops, int* push_ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ops) return;
    push_ok[tid] = s->push(tid) ? 1 : 0;
}

__global__ void stack_pop_kernel(Stack* s, int n_ops, int* pop_out, int* pop_ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ops) return;
    int val = -1;
    bool ok = s->pop(&val);
    pop_ok[tid]  = ok ? 1 : 0;
    pop_out[tid] = ok ? val : -1;
}

// =====================================================
//                   ОЧЕРЕДЬ (GPU, базовая)
// =====================================================
struct Queue {
    int* data;
    int  head;
    int  tail;
    int  capacity;

    __device__ void init(int* buffer, int size) {
        data = buffer;
        head = 0;
        tail = 0;
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        } else {
            atomicSub(&tail, 1);
            return false;
        }
    }

    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(&head, 1);
        int t = tail; // в нашем тесте dequeue запускается после enqueue
        if (pos < t && pos < capacity) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

__global__ void queue_init_kernel(Queue* q, int* buf, int cap) {
    if (blockIdx.x == 0 && threadIdx.x == 0) q->init(buf, cap);
}
__global__ void queue_enqueue_kernel(Queue* q, int n_ops, int* enq_ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ops) return;
    enq_ok[tid] = q->enqueue(tid) ? 1 : 0;
}
__global__ void queue_dequeue_kernel(Queue* q, int n_ops, int* deq_out, int* deq_ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ops) return;
    int val = -1;
    bool ok = q->dequeue(&val);
    deq_ok[tid]  = ok ? 1 : 0;
    deq_out[tid] = ok ? val : -1;
}

// =====================================================
// ДОП. ЗАДАНИЕ 1: MPMC очередь (GPU)
// (несколько производителей/потребителей; синхронизация атомиками)
// =====================================================
struct MpmcQueue {
    int* data;
    int  head;
    int  tail;
    int  capacity;

    __device__ void init(int* buffer, int size) {
        data = buffer;
        head = 0;
        tail = 0;
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        } else {
            atomicSub(&tail, 1);
            return false;
        }
    }

    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(&head, 1);
        int t = tail;
        if (pos < t && pos < capacity) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

__global__ void mpmc_init_kernel(MpmcQueue* q, int* buf, int cap) {
    if (blockIdx.x == 0 && threadIdx.x == 0) q->init(buf, cap);
}
__global__ void mpmc_enqueue_kernel(MpmcQueue* q, int n_ops, int* ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ops) return;
    ok[tid] = q->enqueue(tid) ? 1 : 0;
}
__global__ void mpmc_dequeue_kernel(MpmcQueue* q, int n_ops, int* out, int* ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ops) return;
    int val = -1;
    bool success = q->dequeue(&val);
    ok[tid]  = success ? 1 : 0;
    out[tid] = success ? val : -1;
}

// =====================================================
// ДОП. ЗАДАНИЕ 2: очередь с использованием shared memory (GPU)
// Идея: чтение элемента из global -> в shared -> затем чтение из shared
// =====================================================
struct SharedQueue {
    int* data;
    int  head;
    int  tail;
    int  capacity;

    __device__ void init(int* buffer, int size) {
        data = buffer;
        head = 0;
        tail = 0;
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        } else {
            atomicSub(&tail, 1);
            return false;
        }
    }
};

__global__ void sharedq_init_kernel(SharedQueue* q, int* buf, int cap) {
    if (blockIdx.x == 0 && threadIdx.x == 0) q->init(buf, cap);
}
__global__ void sharedq_enqueue_kernel(SharedQueue* q, int n_ops, int* ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ops) return;
    ok[tid] = q->enqueue(tid) ? 1 : 0;
}

// dequeue с shared memory (размер shared = blockDim.x)
__global__ void sharedq_dequeue_kernel(SharedQueue* q, int n_ops, int* out, int* ok) {
    extern __shared__ int s_buf[];  // shared memory динамически: blockDim.x * sizeof(int)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ops) return;

    int localTid = threadIdx.x;

    // атомарно получаем позицию
    int pos = atomicAdd(&(q->head), 1);
    int t = q->tail; // dequeue запускается после enqueue

    if (pos < t && pos < q->capacity) {
        // копируем из global в shared
        s_buf[localTid] = q->data[pos];
        __syncthreads();

        // читаем из shared
        out[tid] = s_buf[localTid];
        ok[tid]  = 1;
        return;
    }

    ok[tid]  = 0;
    out[tid] = -1;
}

// =====================================================
// ДОП. ЗАДАНИЕ 3: CPU (последовательные версии) + время
// =====================================================
static float ms_since(const std::chrono::high_resolution_clock::time_point& a,
                      const std::chrono::high_resolution_clock::time_point& b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

struct CpuStack {
    std::vector<int> data;
    int top;
    CpuStack(int cap) : data(cap), top(-1) {}

    bool push(int v) {
        int pos = top + 1;
        if (pos < (int)data.size()) {
            top = pos;
            data[pos] = v;
            return true;
        }
        return false;
    }
    bool pop(int& v) {
        if (top >= 0) {
            v = data[top];
            top--;
            return true;
        }
        return false;
    }
};

struct CpuQueue {
    std::vector<int> data;
    int head, tail;
    CpuQueue(int cap) : data(cap), head(0), tail(0) {}

    bool enqueue(int v) {
        if (tail < (int)data.size()) {
            data[tail++] = v;
            return true;
        }
        return false;
    }
    bool dequeue(int& v) {
        if (head < tail) {
            v = data[head++];
            return true;
        }
        return false;
    }
};

// =====================================================
//                         MAIN
// =====================================================
int main() {
    const int N = 1 << 20;       // количество операций
    const int CAPACITY = N;
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    // ---------------------------
    // CPU TIMING (2 времени)
    // ---------------------------
    float cpu_stack_ms = 0.0f, cpu_queue_ms = 0.0f;

    {
        CpuStack st(CAPACITY);
        std::vector<int> okPush(N), okPop(N);
        std::vector<int> out(N);

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) okPush[i] = st.push(i) ? 1 : 0;
        for (int i = 0; i < N; i++) { int v=-1; okPop[i] = st.pop(v) ? 1 : 0; out[i]=v; }
        auto t2 = std::chrono::high_resolution_clock::now();
        cpu_stack_ms = ms_since(t1, t2);
    }

    {
        CpuQueue q(CAPACITY);
        std::vector<int> okEnq(N), okDeq(N);
        std::vector<int> out(N);

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) okEnq[i] = q.enqueue(i) ? 1 : 0;
        for (int i = 0; i < N; i++) { int v=-1; okDeq[i] = q.dequeue(v) ? 1 : 0; out[i]=v; }
        auto t2 = std::chrono::high_resolution_clock::now();
        cpu_queue_ms = ms_since(t1, t2);
    }

    // ---------------------------
    // GPU: STACK
    // ---------------------------
    int* d_stack_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stack_buffer, CAPACITY * sizeof(int)));
    Stack* d_stack = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stack, sizeof(Stack)));
    stack_init_kernel<<<1, 1>>>(d_stack, d_stack_buffer, CAPACITY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_stack_push_ok=nullptr, *d_stack_pop_ok=nullptr, *d_stack_pop_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_stack_push_ok, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stack_pop_ok,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stack_pop_out, N * sizeof(int)));

    cudaEvent_t s1, s2;
    CUDA_CHECK(cudaEventCreate(&s1));
    CUDA_CHECK(cudaEventCreate(&s2));
    CUDA_CHECK(cudaEventRecord(s1));
    stack_push_kernel<<<GRID, BLOCK>>>(d_stack, N, d_stack_push_ok);
    stack_pop_kernel <<<GRID, BLOCK>>>(d_stack, N, d_stack_pop_out, d_stack_pop_ok);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(s2));
    CUDA_CHECK(cudaEventSynchronize(s2));
    float stack_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&stack_ms, s1, s2));

    // ---------------------------
    // GPU: QUEUE (базовая)
    // ---------------------------
    int* d_queue_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_queue_buffer, CAPACITY * sizeof(int)));
    Queue* d_queue = nullptr;
    CUDA_CHECK(cudaMalloc(&d_queue, sizeof(Queue)));
    queue_init_kernel<<<1, 1>>>(d_queue, d_queue_buffer, CAPACITY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_enq_ok=nullptr, *d_deq_ok=nullptr, *d_deq_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_enq_ok,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_deq_ok,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_deq_out, N * sizeof(int)));

    cudaEvent_t q1, q2;
    CUDA_CHECK(cudaEventCreate(&q1));
    CUDA_CHECK(cudaEventCreate(&q2));
    CUDA_CHECK(cudaEventRecord(q1));
    queue_enqueue_kernel<<<GRID, BLOCK>>>(d_queue, N, d_enq_ok);
    queue_dequeue_kernel<<<GRID, BLOCK>>>(d_queue, N, d_deq_out, d_deq_ok);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(q2));
    CUDA_CHECK(cudaEventSynchronize(q2));
    float queue_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&queue_ms, q1, q2));

    // ---------------------------
    // GPU: MPMC QUEUE (доп. задание 1)
    // ---------------------------
    int* d_mpmc_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mpmc_buffer, CAPACITY * sizeof(int)));
    MpmcQueue* d_mpmc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mpmc, sizeof(MpmcQueue)));
    mpmc_init_kernel<<<1, 1>>>(d_mpmc, d_mpmc_buffer, CAPACITY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_mpmc_enq_ok=nullptr, *d_mpmc_deq_ok=nullptr, *d_mpmc_deq_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_mpmc_enq_ok,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mpmc_deq_ok,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mpmc_deq_out, N * sizeof(int)));

    cudaEvent_t m1, m2;
    CUDA_CHECK(cudaEventCreate(&m1));
    CUDA_CHECK(cudaEventCreate(&m2));
    CUDA_CHECK(cudaEventRecord(m1));
    mpmc_enqueue_kernel<<<GRID, BLOCK>>>(d_mpmc, N, d_mpmc_enq_ok);
    mpmc_dequeue_kernel<<<GRID, BLOCK>>>(d_mpmc, N, d_mpmc_deq_out, d_mpmc_deq_ok);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(m2));
    CUDA_CHECK(cudaEventSynchronize(m2));
    float mpmc_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&mpmc_ms, m1, m2));

    // ---------------------------
    // GPU: SHARED QUEUE (доп. задание 2)
    // ---------------------------
    int* d_sharedq_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sharedq_buffer, CAPACITY * sizeof(int)));
    SharedQueue* d_sharedq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sharedq, sizeof(SharedQueue)));
    sharedq_init_kernel<<<1, 1>>>(d_sharedq, d_sharedq_buffer, CAPACITY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_shared_enq_ok=nullptr, *d_shared_deq_ok=nullptr, *d_shared_deq_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_shared_enq_ok,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_shared_deq_ok,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_shared_deq_out, N * sizeof(int)));

    cudaEvent_t sh1, sh2;
    CUDA_CHECK(cudaEventCreate(&sh1));
    CUDA_CHECK(cudaEventCreate(&sh2));
    CUDA_CHECK(cudaEventRecord(sh1));
    sharedq_enqueue_kernel<<<GRID, BLOCK>>>(d_sharedq, N, d_shared_enq_ok);
    // shared memory size = BLOCK * sizeof(int)
    sharedq_dequeue_kernel<<<GRID, BLOCK, BLOCK * sizeof(int)>>>(d_sharedq, N, d_shared_deq_out, d_shared_deq_ok);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(sh2));
    CUDA_CHECK(cudaEventSynchronize(sh2));
    float sharedq_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&sharedq_ms, sh1, sh2));

    // ---------------------------
    // ВЫВОД (6 времен: 2 CPU + 4 GPU)
    // ---------------------------
    printf("\n=============================\n");
    printf("   RESULTS (CPU vs GPU CUDA)\n");
    printf("=============================\n");

    printf("\n[CPU SEQUENTIAL]\n");
    printf("CPU Stack time         : %.3f ms\n", cpu_stack_ms);
    printf("CPU Queue time         : %.3f ms\n", cpu_queue_ms);

    printf("\n[GPU CUDA]\n");
    printf("GPU Stack time         : %.3f ms\n", stack_ms);
    printf("GPU Queue (basic) time : %.3f ms\n", queue_ms);
    printf("GPU Queue (MPMC) time  : %.3f ms\n", mpmc_ms);
    printf("GPU Queue (shared) time: %.3f ms\n", sharedq_ms);

    printf("\n[COMPARE]\n");
    if (queue_ms > 0.0f) printf("stack/queue ratio      : %.2f x\n", stack_ms / queue_ms);
    printf("Note: время зависит от GPU, размера блока и конкуренции атомарных операций.\n");

    // ---------------------------
    // CLEANUP
    // ---------------------------
    CUDA_CHECK(cudaEventDestroy(s1)); CUDA_CHECK(cudaEventDestroy(s2));
    CUDA_CHECK(cudaEventDestroy(q1)); CUDA_CHECK(cudaEventDestroy(q2));
    CUDA_CHECK(cudaEventDestroy(m1)); CUDA_CHECK(cudaEventDestroy(m2));
    CUDA_CHECK(cudaEventDestroy(sh1)); CUDA_CHECK(cudaEventDestroy(sh2));

    CUDA_CHECK(cudaFree(d_stack_buffer));
    CUDA_CHECK(cudaFree(d_stack));
    CUDA_CHECK(cudaFree(d_stack_push_ok));
    CUDA_CHECK(cudaFree(d_stack_pop_ok));
    CUDA_CHECK(cudaFree(d_stack_pop_out));

    CUDA_CHECK(cudaFree(d_queue_buffer));
    CUDA_CHECK(cudaFree(d_queue));
    CUDA_CHECK(cudaFree(d_enq_ok));
    CUDA_CHECK(cudaFree(d_deq_ok));
    CUDA_CHECK(cudaFree(d_deq_out));

    CUDA_CHECK(cudaFree(d_mpmc_buffer));
    CUDA_CHECK(cudaFree(d_mpmc));
    CUDA_CHECK(cudaFree(d_mpmc_enq_ok));
    CUDA_CHECK(cudaFree(d_mpmc_deq_ok));
    CUDA_CHECK(cudaFree(d_mpmc_deq_out));

    CUDA_CHECK(cudaFree(d_sharedq_buffer));
    CUDA_CHECK(cudaFree(d_sharedq));
    CUDA_CHECK(cudaFree(d_shared_enq_ok));
    CUDA_CHECK(cudaFree(d_shared_deq_ok));
    CUDA_CHECK(cudaFree(d_shared_deq_out));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
