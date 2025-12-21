#include <iostream>
#include <cstdlib> // Для rand(), srand(), NULL
#include <ctime>   // Для time()
#include <omp.h>   // Для OpenMP

// 2. Функция для поиска среднего значения элементов массива
// Использует указатель на начало массива (int* arr) и его размер.
double calculate_average(int* arr, int size) {
    // Проверка на пустой массив
    if (size <= 0) {
        return 0.0;
    }

    // Инициализация суммы, должна быть double для точного среднего
    double sum = 0.0;

    // Директива: #pragma omp parallel for reduction(+:sum)
    // Она безопасно распределяет работу по сложению между несколькими потоками.
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }

    // Возвращаем среднее значение
    return sum / size;
} // <-- ВАЖНО: Закрывающая скобка для функции calculate_average!

int main() {
    // Устанавливаем размер массива
    const int N = 1000000000; 

    // Инициализация генератора случайных чисел
    srand(time(0));

    // --- 1. Создание динамического массива с помощью указателей ---
    int* array_ptr = nullptr;
    try {
        // Выделение непрерывного блока памяти в куче (heap)
        array_ptr = new int[N]; 
        std::cout << "1. Динамический массив размером " << N << " создан." << std::endl;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Ошибка выделения памяти: " << e.what() << std::endl;
        return 1; // Выход с ошибкой
    }

    // Заполнение массива случайными числами (от 0 до 99)
    for (int i = 0; i < N; ++i) {
        array_ptr[i] = rand() % 100;
    }
    std::cout << "   Массив заполнен случайными числами." << std::endl;

    // --- 3. Параллельный подсчёт среднего значения ---
    std::cout << "3. Начинаем параллельный подсчёт среднего значения..." << std::endl;

    // Измерение времени начала
    double start_time = omp_get_wtime();

    double average = calculate_average(array_ptr, N);

    // Измерение времени окончания
    double end_time = omp_get_wtime();

    // Вывод результатов
    std::cout << "   Среднее значение элементов массива: " << average << std::endl;
    std::cout << "   Время выполнения (OpenMP): " << (end_time - start_time) * 1000 << " мс." << std::endl;

    // --- 4. Освобождение динамической памяти ---
    if (array_ptr != nullptr) {
        delete[] array_ptr; // Используется delete[] для массивов!
        array_ptr = nullptr; 
        std::cout << "4. Динамическая память успешно освобождена (delete[])." << std::endl;
    }

    #ifdef _WIN32 
    std::cout << "\nНажмите Enter для выхода..." << std::endl;
    std::cin.get();
    #endif

    return 0;
} 