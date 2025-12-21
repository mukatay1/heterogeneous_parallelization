#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>    // Библиотека OpenMP

int main() {
    const int size = 1000000;
    std::vector<int> data(size);

    // Заполнение массива
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-1000000, 1000000);
    for (int i = 0; i < size; ++i) data[i] = dist(gen);

    int min_val, max_val;

    // Паралелльный поиск
    auto start = std::chrono::high_resolution_clock::now();

    min_val = data[0];
    max_val = data[0];

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < size; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> parallel_time = end - start;

    std::cout << "Минимум: " << min_val << ", Максимум: " << max_val << std::endl;
    std::cout << "Время: " << parallel_time.count() << " мс" << std::endl;

    return 0;
}