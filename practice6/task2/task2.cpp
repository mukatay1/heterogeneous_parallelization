#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>

static std::string loadTextFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

static void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error " << err << " at: " << what << "\n";
        std::exit(1);
    }
}

// Последовательная проверка на CPU
static void matmul_cpu(const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C,
                       int N, int M, int K)
{
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < K; c++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                sum += A[r * M + i] * B[i * K + c];
            }
            C[r * K + c] = sum;
        }
    }
}

int main() {
    // ===== Размеры матриц =====
    // Можно менять под себя
    const int N = 128; // строки A и C
    const int M = 256; // столбцы A и строки B
    const int K = 64;  // столбцы B и C

    const size_t sizeA = (size_t)N * M;
    const size_t sizeB = (size_t)M * K;
    const size_t sizeC = (size_t)N * K;

    // ===== Инициализация данных =====
    std::vector<float> A(sizeA), B(sizeB), C(sizeC, 0.0f), Cref(sizeC, 0.0f);

    // Заполняем простыми значениями (чтобы не было NaN и легче проверять)
    for (size_t i = 0; i < sizeA; i++) A[i] = (float)((i % 13) + 1);
    for (size_t i = 0; i < sizeB; i++) B[i] = (float)((i % 7) + 1);

    // ===== OpenCL: платформа/устройство =====
    cl_int err;
    cl_platform_id platform = nullptr;
    check(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");

    // Пробуем GPU, если нет — CPU
    cl_device_id device = nullptr;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr),
              "clGetDeviceIDs(CPU fallback)");
    }

    // ===== Контекст и очередь =====
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check(err, "clCreateContext");

    // Для простоты: обычная очередь (без профайлинга)
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check(err, "clCreateCommandQueue");

    // ===== Буферы =====
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * sizeA, (void*)A.data(), &err);
    check(err, "clCreateBuffer(A)");

    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * sizeB, (void*)B.data(), &err);
    check(err, "clCreateBuffer(B)");

    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * sizeC, nullptr, &err);
    check(err, "clCreateBuffer(C)");

    // ===== Программа и ядро =====
    std::string src = loadTextFile("matmul.cl");
    if (src.empty()) {
        std::cerr << "Не найден файл matmul.cl рядом с main.cpp\n";
        return 1;
    }

    const char* srcPtr = src.c_str();
    size_t srcLen = src.size();

    cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Если сборка упала — покажем лог
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        check(err, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    check(err, "clCreateKernel(matmul)");

    // Аргументы ядра
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA), "clSetKernelArg(0)");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB), "clSetKernelArg(1)");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC), "clSetKernelArg(2)");
    check(clSetKernelArg(kernel, 3, sizeof(int), &N), "clSetKernelArg(3)");
    check(clSetKernelArg(kernel, 4, sizeof(int), &M), "clSetKernelArg(4)");
    check(clSetKernelArg(kernel, 5, sizeof(int), &K), "clSetKernelArg(5)");

    // ===== Запуск =====
    // 2D глобальная сетка: (N, K)
    size_t global[2] = { (size_t)N, (size_t)K };

    auto t0 = std::chrono::high_resolution_clock::now();
    check(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");
    check(clFinish(queue), "clFinish");
    auto t1 = std::chrono::high_resolution_clock::now();

    // Читаем C
    check(clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                              sizeof(float) * sizeC, C.data(),
                              0, nullptr, nullptr),
          "clEnqueueReadBuffer(C)");

    std::chrono::duration<double, std::milli> cl_ms = t1 - t0;

    // ===== Проверка корректности (с CPU) =====
    auto c0 = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, Cref, N, M, K);
    auto c1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = c1 - c0;

    // Сравнение с допуском
    int bad = 0;
    for (size_t i = 0; i < sizeC; i++) {
        float diff = std::fabs(C[i] - Cref[i]);
        if (diff > 1e-3f) {
            bad++;
            if (bad < 5) {
                std::cerr << "Mismatch at " << i << ": OpenCL=" << C[i]
                          << " CPU=" << Cref[i] << " diff=" << diff << "\n";
            }
        }
    }

    std::cout << "===== Task 2: Matrix Multiplication (OpenCL) =====\n";
    std::cout << "Размеры: A=" << N << "x" << M << ", B=" << M << "x" << K
              << ", C=" << N << "x" << K << "\n\n";

    std::cout << "OpenCL time: " << cl_ms.count() << " ms\n";
    std::cout << "CPU time:    " << cpu_ms.count() << " ms\n";
    std::cout << "Проверка:    " << (bad == 0 ? "OK (совпало)" : "FAILED") << "\n";

    // Для отчёта — покажем пару элементов
    std::cout << "\nПример значений C:\n";
    std::cout << "C[0,0] = " << C[0] << "\n";
    std::cout << "C[N-1,K-1] = " << C[(size_t)(N - 1) * K + (K - 1)] << "\n";

    // ===== Очистка =====
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
