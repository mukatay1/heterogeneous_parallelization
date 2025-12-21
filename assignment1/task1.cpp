#include <iostream>
#include <random>   // Для генерации случайных чисел
#include <numeric>  // Для вычисление суммы

int main() {
    // Динамическое выделение памяти для массива из 50 000 целых чисел
    const int size = 50000;
    int* array = new int[size];

    // Настройка генератора случайных чисел (от 1 до 100)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 100);

    // Заполнение массива случайными значениями
    for (int i = 0; i < size; ++i) {
        array[i] = dist(gen);
    }

    // Вычисление среднего значения
    // Используем long long для суммы, чтобы избежать переполнения
    long long sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }

    double average = static_cast<double>(sum) / size;

    // Вывод результата
    std::cout << "Среднее значение элементов массива: " << average << std::endl;

    // Корректное освобождение памяти
    delete[] array;
    array = nullptr; // Хорошая практика: зануление указателя

    return 0;
}