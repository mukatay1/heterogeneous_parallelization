#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <random>

using namespace std;

// 1. Параллельная сортировка пузырьком 
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        // Четная фаза
        if (i % 2 == 0) {
            #pragma omp parallel for
            for (int j = 1; j < n; j += 2) {
                if (arr[j - 1] > arr[j]) swap(arr[j - 1], arr[j]);
            }
        } 
        // Нечетная фаза
        else {
            #pragma omp parallel for
            for (int j = 2; j < n; j += 2) {
                if (arr[j - 1] > arr[j]) swap(arr[j - 1], arr[j]);
            }
        }
    }
}

// 2. Параллельная сортировка выбором
void parallelSelectionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_val = arr[i];
        int min_idx = i;

        // Распараллеливаем только поиск минимального элемента
        #pragma omp parallel
        {
            int local_min = min_val;
            int local_idx = min_idx;

            #pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < local_min) {
                    local_min = arr[j];
                    local_idx = j;
                }
            }

            #pragma omp critical
            {
                if (local_min < min_val) {
                    min_val = local_min;
                    min_idx = local_idx;
                }
            }
        }
        swap(arr[i], arr[min_idx]);
    }
}

// 3. Сортировка вставкой 
// Примечание: Эффективность низкая из-за зависимостей
void parallelInsertionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        
        // Здесь используется последовательный подход внутри параллельной секции
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void testSort(string name, void (*sortFunc)(vector<int>&), int size) {
    vector<int> data(size);
    mt19937 gen(42);
    uniform_int_distribution<> dist(1, 100000);
    for (int& x : data) x = dist(gen);

    auto start = chrono::high_resolution_clock::now();
    sortFunc(data);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> diff = end - start;
    cout << name << " (" << size << " эл.): " << diff.count() << " ms" << endl;
}

int main() {
    // Устанавливаем количество потоков (например, 4)
    omp_set_num_threads(4);
    
    int sizes[] = {1000, 10000, 100000};

    for (int size : sizes) {
        cout << "--- Тестирование для размера " << size << " ---" << endl;
        testSort("Bubble (Parallel)", parallelBubbleSort, size);
        testSort("Selection (Parallel)", parallelSelectionSort, size);
        testSort("Insertion (Parallel)", parallelInsertionSort, size);
        cout << endl;
    }

    return 0;
}