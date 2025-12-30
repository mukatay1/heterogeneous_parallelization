#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// Последовательная сортировка выбором
void selectionSort(vector<int> arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        swap(arr[i], arr[min_idx]);
    }
}

// Параллельная сортировка выбором
void parallelSelectionSort(vector<int> arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_val = arr[i + 1];
        int min_idx = i + 1;

        // Параллельный поиск минимума в оставшейся части массива
        #pragma omp parallel
        {
            int local_min = arr[i + 1];
            int local_idx = i + 1;

            #pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < local_min) {
                    local_min = arr[j];
                    local_idx = j;
                }
            }

            // Критическая секция для обновления глобального минимума
            #pragma omp critical
            {
                if (local_min < min_val) {
                    min_val = local_min;
                    min_idx = local_idx;
                }
            }
        }

        if (min_val < arr[i]) {
            swap(arr[i], arr[min_idx]);
        }
    }
}

void runTest(int size) {
    vector<int> data(size);
    for (int i = 0; i < size; i++) data[i] = rand() % size;

    cout << "\nТест для массива из " << size << " элементов " << endl;

    // Замер последовательной
    auto start = high_resolution_clock::now();
    selectionSort(data);
    auto end = high_resolution_clock::now();
    cout << "Последовательная: " << duration<double, milli>(end - start).count() << " ms" << endl;

    // Замер параллельной
    start = high_resolution_clock::now();
    parallelSelectionSort(data);
    end = high_resolution_clock::now();
    cout << "Параллельная (OpenMP): " << duration<double, milli>(end - start).count() << " ms" << endl;
}

int main() {
    runTest(1000);
    runTest(10000);
    return 0;
}