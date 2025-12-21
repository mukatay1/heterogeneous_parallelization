#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <random>
#include <iomanip>

using namespace std;


void bubbleSortSeq(vector<int> arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
}

void selectionSortSeq(vector<int> arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++)
            if (arr[j] < arr[min_idx]) min_idx = j;
        swap(arr[i], arr[min_idx]);
    }
}


void bubbleSortPar(vector<int> arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            #pragma omp parallel for
            for (int j = 1; j < n; j += 2)
                if (arr[j - 1] > arr[j]) swap(arr[j - 1], arr[j]);
        } else {
            #pragma omp parallel for
            for (int j = 2; j < n; j += 2)
                if (arr[j - 1] > arr[j]) swap(arr[j - 1], arr[j]);
        }
    }
}

void selectionSortPar(vector<int> arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        int min_val = arr[i];
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


void run_test(int size) {
    vector<int> base_data(size);
    mt19937 gen(42);
    uniform_int_distribution<> dist(1, 10000);
    for (int& x : base_data) x = dist(gen);

    cout << "\nРАЗМЕР МАССИВА: " << size << endl;
    cout << left << setw(25) << "Алгоритм" << setw(15) << "Время (мс)" << endl;
    cout << "---------------------------------------------" << endl;

    auto benchmark = [&](string name, void (*func)(vector<int>)) {
        auto start = chrono::high_resolution_clock::now();
        func(base_data);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = end - start;
        cout << left << setw(25) << name << setw(15) << fixed << setprecision(2) << elapsed.count() << endl;
    };

    benchmark("Bubble Sequential", bubbleSortSeq);
    benchmark("Bubble Parallel", bubbleSortPar);
    benchmark("Selection Sequential", selectionSortSeq);
    benchmark("Selection Parallel", selectionSortPar);
}

int main() {
    omp_set_num_threads(omp_get_max_threads());
    run_test(1000);
    run_test(10000);
    run_test(50000); 

    return 0;
}