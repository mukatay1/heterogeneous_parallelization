#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <iomanip>

// --------- Утилита: строка -> lower ---------
static std::string lower(std::string s) {
    for (char &c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// --------- Утилита: чтение long long ---------
static long long to_ll(const char* s, long long def) {
    if (!s) return def;
    long long v = std::atoll(s);
    return (v > 0) ? v : def;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Аргументы:
    // argv[1] mode: strong | weak
    // argv[2] op: sum | min | max
    // argv[3] comm: reduce | allreduce
    // argv[4] N: для strong это global N, для weak это local N на процесс
    std::string mode = (argc >= 2) ? lower(argv[1]) : "strong";
    std::string op   = (argc >= 3) ? lower(argv[2]) : "sum";
    std::string comm = (argc >= 4) ? lower(argv[3]) : "reduce";
    long long N_arg  = (argc >= 5) ? to_ll(argv[4], 20000000LL) : 20000000LL;

    bool isStrong = (mode == "strong");
    bool useAllreduce = (comm == "allreduce");

    // ----- Определяем размер локального массива -----
    // Strong scaling: глобальный N фиксирован, делим между процессами
    // Weak scaling: N_local фиксирован, глобальный растёт как N_local * P
    long long N_global = 0;
    long long N_local  = 0;

    if (isStrong) {
        N_global = N_arg;
        N_local  = N_global / size;         // простое деление
        long long rem = N_global % size;    // остаток
        // распределим остаток: первые rem процессов получат +1 элемент
        if (rank < rem) N_local++;
    } else {
        N_local  = N_arg;                   // фиксировано на процесс
        N_global = N_local * (long long)size;
    }

    // ----- Генерируем локальные данные -----
    // Важно: именно локальный массив, без пересылок (для масштабируемости)
    std::vector<double> local((size_t)N_local);

    std::mt19937_64 gen(12345ULL + (unsigned long long)rank);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (long long i = 0; i < N_local; i++) local[(size_t)i] = dist(gen);

    // ----- Локальная агрегация (полезная работа) -----
    // Для sum: local_sum
    // Для min/max: local_min/local_max
    double local_val = 0.0;

    // Засекаем "полное время" (локальное вычисление + коммуникация)
    MPI_Barrier(MPI_COMM_WORLD); // синхронизация перед измерением
    double t0 = MPI_Wtime();

    if (op == "sum") {
        double s = 0.0;
        for (long long i = 0; i < N_local; i++) s += local[(size_t)i];
        local_val = s;
    } else if (op == "min") {
        double m = local.empty() ? 0.0 : local[0];
        for (long long i = 1; i < N_local; i++) m = std::min(m, local[(size_t)i]);
        local_val = m;
    } else if (op == "max") {
        double m = local.empty() ? 0.0 : local[0];
        for (long long i = 1; i < N_local; i++) m = std::max(m, local[(size_t)i]);
        local_val = m;
    } else {
        if (rank == 0) std::cerr << "Неизвестная операция op. Используй: sum|min|max\n";
        MPI_Finalize();
        return 1;
    }

    // ----- Коммуникационная часть: Reduce или Allreduce -----
    MPI_Op mpi_op = MPI_SUM;
    if (op == "min") mpi_op = MPI_MIN;
    if (op == "max") mpi_op = MPI_MAX;

    double global_val = 0.0;

    if (useAllreduce) {
        // Все процессы получают результат
        MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, mpi_op, MPI_COMM_WORLD);
    } else {
        // Результат только на rank 0
        MPI_Reduce(&local_val, &global_val, 1, MPI_DOUBLE, mpi_op, 0, MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;

    // Для корректного отчёта берём максимум времени среди процессов
    // (в распределённых программах именно самый медленный процесс определяет время итерации)
    double time_max = 0.0;
    MPI_Reduce(&local_time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ----- Печать результатов -----
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);

        std::cout << "Практическая №10 / Таск 4: Масштабируемость MPI\n";
        std::cout << "Mode: " << (isStrong ? "strong" : "weak")
                  << ", op: " << op
                  << ", comm: " << (useAllreduce ? "MPI_Allreduce" : "MPI_Reduce")
                  << "\n";

        std::cout << "Processes P = " << size
                  << ", N_global = " << N_global
                  << ", N_local(avg) ~ " << (N_global / size)
                  << "\n";

        std::cout << "Time (max over ranks) = " << time_max << " seconds\n";

        // Показать агрегатный результат (для контроля)
        // Для Reduce он есть только на root; для Allreduce он есть у всех
        std::cout << "Aggregate result = " << global_val << "\n";
    }

    MPI_Finalize();
    return 0;
}
