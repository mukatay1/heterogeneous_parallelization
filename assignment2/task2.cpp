#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

int main() {
    const int SIZE = 10000;
    vector<int> data(SIZE);

    // 1. Инициализация массива случайными числами
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100000);

    for (int i = 0; i < SIZE; ++i) {
        data[i] = dis(gen);
    }

    int min_val, max_val;

    // 2. Последовательная реализация
    auto start_seq = high_resolution_clock::now();
    
    min_val = data[0];
    max_val = data[0];
    for (int i = 1; i < SIZE; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    auto end_seq = high_resolution_clock::now();
    duration<double, milli> diff_seq = end_seq - start_seq;

    cout << "Последовательно: min = " << min_val << ", max = " << max_val << endl;
    cout << "Время (seq): " << diff_seq.count() << " ms" << endl;

    // 3. Параллельная реализация (OpenMP)
    auto start_par = high_resolution_clock::now();
    
    int p_min = data[0];
    int p_max = data[0];

    #pragma omp parallel for reduction(min:p_min) reduction(max:p_max)
    for (int i = 0; i < SIZE; ++i) {
        if (data[i] < p_min) p_min = data[i];
        if (data[i] > p_max) p_max = data[i];
    }

    auto end_par = high_resolution_clock::now();
    duration<double, milli> diff_par = end_par - start_par;

    cout << "Параллельно (OpenMP): min = " << p_min << ", max = " << p_max << endl;
    cout << "Время (OMP): " << diff_par.count() << " ms" << endl;

    return 0;
}