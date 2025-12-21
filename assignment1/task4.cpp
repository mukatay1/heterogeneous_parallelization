#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>
#include <omp.h>

int main() {
    const int size = 5000000;
    std::vector<int> data(size);

    // Подготовка данных
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 100);
    for (int i = 0; i < size; ++i) data[i] = dist(gen);

    // последовательное вычисление
    auto start_seq = std::chrono::high_resolution_clock::now();
    
    long long sum_seq = 0;
    for (int i = 0; i < size; ++i) {
        sum_seq += data[i];
    }
    double avg_seq = static_cast<double>(sum_seq) / size;
    
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_seq = end_seq - start_seq;

    // ПАРАЛЛЕЛЬНОЕ вычисление (OpenMP)
    auto start_par = std::chrono::high_resolution_clock::now();
    
    long long sum_par = 0;
    //  суммирует локальные результаты каждого потока в общую переменную
    #pragma omp parallel for reduction(+:sum_par)
    for (int i = 0; i < size; ++i) {
        sum_par += data[i];
    }
    double avg_par = static_cast<double>(sum_par) / size;
    
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_par = end_par - start_par;

    // Вывод результатов
    std::cout << "Последовательно: " << avg_seq << " | Время: " << time_seq.count() << " мс" << std::endl;
    std::cout << "Параллельно (OMP): " << avg_par << " | Время: " << time_par.count() << " мс" << std::endl;
    std::cout << "Ускорение: " << time_seq.count() / time_par.count() << "x" << std::endl;

    return 0;
}