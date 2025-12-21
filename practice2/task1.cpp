#include <iostream>
#include <vector>
#include <algorithm> // Для std::swap

// 1. Сортировка пузырьком 
// Сравнивает соседние элементы и "выталкивает" самые большие в конец
void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// 2. Сортировка выбором 
// Находит минимальный элемент в остатке массива и ставит его на текущее место
void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < n; ++j) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        std::swap(arr[i], arr[min_idx]);
    }
}

// 3. Сортировка вставкой 
// Берет элемент и "вставляет" его в правильную позицию в уже отсортированной левой части
void insertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;
        // Сдвигаем элементы, которые больше ключа, на одну позицию вперед
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

// Вспомогательная функция для вывода массива
void printArray(const std::vector<int>& arr) {
    for (int x : arr) std::cout << x << " ";
    std::cout << std::endl;
}

int main() {
    std::vector<int> data = {64, 25, 12, 22, 11};

    std::cout << "Исходный массив: ";
    printArray(data);

    // Пример вызова сортировки выбором
    selectionSort(data);

    std::cout << "Отсортированный массив: ";
    printArray(data);

    return 0;
}