// ==============================
// Practical Work 7 — Task 3
// Анализ производительности (время vs размер массива)
// Для редукции (sum) и сканирования (prefix sum)
// ВАРИАНТ "как раньше": запускается в CMD без CUDA,
// но выводит таблицу CPU/GPU (GPU — эмуляция) + данные для графика.
// ==============================

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <string>
#include <cmath>

#ifdef _WIN32
  #include <windows.h>
#endif

// ------------------------------
// CPU редукция (сумма массива)
// ------------------------------
static double cpu_reduce_sum(const std::vector<float>& a) {
    // std::accumulate — последовательная сумма
    return std::accumulate(a.begin(), a.end(), 0.0);
}

// ------------------------------
// CPU scan (inclusive prefix sum)
// out[i] = a[0] + ... + a[i]
// ------------------------------
static void cpu_prefix_scan(const std::vector<float>& a, std::vector<float>& out) {
    if (a.empty()) return;
    out.resize(a.size());
    out[0] = a[0];
    for (size_t i = 1; i < a.size(); i++) {
        out[i] = out[i - 1] + a[i];
    }
}

// ------------------------------
// Эмуляция GPU времени
// Идея: GPU обычно быстрее на больших размерах,
// но есть накладные расходы (копирование/запуск).
// Мы делаем простую модель:
// gpu_ms = overhead_ms + cpu_ms / speedup
// ------------------------------
static double emulate_gpu_time(double cpu_ms, double overhead_ms, double speedup) {
    return overhead_ms + cpu_ms / speedup;
}

int main() {
#ifdef _WIN32
    // Чтобы русский текст нормально отображался в Windows CMD
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    // ===== Размеры массивов для теста (как в методичке) =====
    // Можно менять/добавлять свои размеры
    std::vector<size_t> sizes = {
        1024,
        1000000,
        10000000
    };

    // ===== Параметры эмуляции GPU =====
    // overhead — накладные расходы (копии + запуск ядра)
    // speedup  — ускорение относительно CPU
    const double overhead_reduce_ms = 0.05;  // редукция обычно проще
    const double overhead_scan_ms   = 0.08;  // сканирование обычно сложнее
    const double speedup_reduce     = 8.0;
    const double speedup_scan       = 6.0;

    // ===== Заголовок =====
    std::cout << "===== Practical Work 7 / Task 3: Performance Analysis =====\n";
    std::cout << "Операции: Reduction (sum) и Scan (prefix sum)\n\n";

    // ===== Табличный вывод =====
    std::cout << "Таблица результатов (ms):\n";
    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(18) << "CPU_reduce"
              << std::setw(18) << "GPU_reduce"
              << std::setw(18) << "CPU_scan"
              << std::setw(18) << "GPU_scan"
              << "\n";

    std::cout << std::string(84, '-') << "\n";

    // ===== Дополнительно: CSV для графика =====
    // Можно скопировать эти строки в Excel и построить графики
    std::cout << "\nCSV для графика (копируй в Excel):\n";
    std::cout << "N,cpu_reduce_ms,gpu_reduce_ms,cpu_scan_ms,gpu_scan_ms\n";

    // ===== Прогоны по размерам =====
    for (size_t N : sizes) {
        // 1) Генерируем тестовый массив
        // Чтобы результат был предсказуемый и проверяемый — заполняем 1.0
        std::vector<float> a(N, 1.0f);
        std::vector<float> scan_out;

        // ------------------------------
        // CPU: Reduction
        // ------------------------------
        auto t1 = std::chrono::high_resolution_clock::now();
        double sum_cpu = cpu_reduce_sum(a);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_reduce_ms = t2 - t1;

        // ------------------------------
        // CPU: Scan
        // ------------------------------
        auto t3 = std::chrono::high_resolution_clock::now();
        cpu_prefix_scan(a, scan_out);
        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_scan_ms = t4 - t3;

        // ------------------------------
        // GPU (эмуляция времени)
        // ------------------------------
        double gpu_reduce_ms = emulate_gpu_time(cpu_reduce_ms.count(), overhead_reduce_ms, speedup_reduce);
        double gpu_scan_ms   = emulate_gpu_time(cpu_scan_ms.count(),   overhead_scan_ms,   speedup_scan);

        // ------------------------------
        // Быстрая "проверка корректности" результатов CPU
        // sum должен быть N, scan_last должен быть N (если массив из 1.0)
        // ------------------------------
        bool ok_sum = std::fabs(sum_cpu - (double)N) < 1e-6;
        bool ok_scan = (!scan_out.empty()) && (std::fabs(scan_out.back() - (float)N) < 1e-3f);

        // ------------------------------
        // Печать таблицы
        // ------------------------------
        std::cout << std::left << std::fixed << std::setprecision(6)
                  << std::setw(12) << N
                  << std::setw(18) << cpu_reduce_ms.count()
                  << std::setw(18) << gpu_reduce_ms
                  << std::setw(18) << cpu_scan_ms.count()
                  << std::setw(18) << gpu_scan_ms
                  << "\n";

        // CSV строка
        std::cout << N << ","
                  << std::fixed << std::setprecision(6)
                  << cpu_reduce_ms.count() << ","
                  << gpu_reduce_ms << ","
                  << cpu_scan_ms.count() << ","
                  << gpu_scan_ms << "\n";

        // Пояснение по корректности (чтобы можно было показать в отчёте)
        std::cout << "Проверка (CPU): sum=" << (ok_sum ? "OK" : "FAIL")
                  << ", scan_last=" << (ok_scan ? "OK" : "FAIL") << "\n\n";
    }

    // ===== Итоговый вывод =====
    std::cout << "Вывод:\n";
    std::cout << "- CPU время растёт с увеличением N.\n";
    std::cout << "- GPU (по модели) быстрее на больших N, но имеет overhead.\n";
    std::cout << "- Данные CSV можно использовать для построения графиков.\n";

    return 0;
}
