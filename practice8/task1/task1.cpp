#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    // Размер массива
    const int N = 1000000;

    // Создаём массив и инициализируем его
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = i * 1.0f;
    }

    // Засекаем время начала выполнения
    double start_time = omp_get_wtime();

    // Параллельная обработка массива с использованием OpenMP
    // Каждый элемент массива умножается на 2
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        data[i] *= 2.0f;
    }

    // Засекаем время окончания выполнения
    double end_time = omp_get_wtime();

    // Вывод времени выполнения
    std::cout << "Время выполнения на CPU (OpenMP): "
              << end_time - start_time << " секунд" << std::endl;

    // Проверка корректности (вывод первых 5 элементов)
    std::cout << "Первые элементы массива: ";
    for (int i = 0; i < 5; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
