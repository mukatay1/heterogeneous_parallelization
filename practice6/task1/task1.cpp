#include <CL/cl.h>          // Основной заголовочный файл OpenCL
#include <iostream>        // Ввод / вывод
#include <vector>          // Контейнер vector для массивов
#include <fstream>         // Чтение файла с ядром
#include <chrono>          // Измерение времени выполнения

// Размер массивов (1 миллион элементов)
#define ARRAY_SIZE 1024 * 1024

// Функция загрузки OpenCL-ядра из файла
std::string loadKernel(const char* filename) {
    std::ifstream file(filename);   // Открываем файл с ядром
    // Считываем файл целиком в строку
    return std::string(
        (std::istreambuf_iterator<char>(file)),
         std::istreambuf_iterator<char>()
    );
}

int main() {
    cl_int err;  // Переменная для хранения кодов ошибок OpenCL

    // 1. Получение платформы OpenCL
    // Платформа — это реализация OpenCL (Intel, AMD, NVIDIA и т.д.)
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // 2. Получение устройства
    // Пытаемся сначала получить GPU
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Если GPU недоступен — используем CPU
    if (err != CL_SUCCESS) {
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }

    // 3. Создание контекста и очереди
    // Контекст объединяет устройства и память OpenCL
    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, &err
    );

    // Очередь команд управляет выполнением операций на устройстве
    cl_command_queue queue = clCreateCommandQueue(
        context, device, 0, &err
    );

    // ===============================
    // 4. Подготовка данных
    // ===============================
    // Инициализируем входные массивы
    std::vector<float> A(ARRAY_SIZE, 1.0f);
    std::vector<float> B(ARRAY_SIZE, 2.0f);
    std::vector<float> C(ARRAY_SIZE); // Результирующий массив

    // Создаем буферы в памяти OpenCL
    cl_mem bufA = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * ARRAY_SIZE,
        A.data(),
        &err
    );

    cl_mem bufB = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * ARRAY_SIZE,
        B.data(),
        &err
    );

    cl_mem bufC = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * ARRAY_SIZE,
        nullptr,
        &err
    );

    // 5. Загрузка и компиляция ядра
    std::string source = loadKernel("kernel.cl");
    const char* src = source.c_str();
    size_t size = source.size();

    // Создаем программу OpenCL из исходного кода
    cl_program program = clCreateProgramWithSource(
        context, 1, &src, &size, &err
    );

    // Компилируем программу под выбранное устройство
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Создаем ядро (kernel) из программы
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);

    // 6. Передача аргументов в ядро
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // 7. Запуск ядра
    // Глобальный размер — количество потоков
    size_t globalSize = ARRAY_SIZE;

    // Засекаем время выполнения
    auto start = std::chrono::high_resolution_clock::now();

    // Запускаем ядро на устройстве
    clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0,
        nullptr,
        nullptr
    );

    // Ожидаем завершения выполнения всех команд
    clFinish(queue);

    auto end = std::chrono::high_resolution_clock::now();

    // 8. Чтение результата
    clEnqueueReadBuffer(
        queue,
        bufC,
        CL_TRUE,
        0,
        sizeof(float) * ARRAY_SIZE,
        C.data(),
        0,
        nullptr,
        nullptr
    );

    // Вычисляем время выполнения
    std::chrono::duration<double> elapsed = end - start;

    // Вывод результатов
    std::cout << "Время выполнения: " << elapsed.count() << " сек\n";
    std::cout << "Пример результата: C[0] = " << C[0] << std::endl;

    // 9. Освобождение ресурсов
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
