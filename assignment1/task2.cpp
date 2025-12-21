#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>   // Для замера времени
#include <random>

int main() {
    const int size = 1000000;
    // Используем std::vector для удобного управления памятью 
    std::vector<int> data(size);

    // Заполнение массива случайными числами
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-1000000, 1000000);

    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }

    // Начало алгоритма поиска
    auto start = std::chrono::high_resolution_clock::now();

    int min_val = data[0];
    int max_val = data[0];

    for (int i = 1; i < size; ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    // Конец алгоритма поиска

    // Вычисляем длительность в миллисекундах
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Минимальный элемент: " << min_val << std::endl;
    std::cout << "Максимальный элемент: " << max_val << std::endl;
    std::cout << "Время выполнения: " << duration.count() << " мс" << std::endl;

    return 0;
}