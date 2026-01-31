#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

// Функция для безопасного чтения N из аргументов командной строки
long long readN(int argc, char** argv)
{
    // Если пользователь передал N, используем его, иначе N = 1e6
    if (argc >= 2) return std::atoll(argv[1]);
    return 1000000LL;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // сколько всего процессов

    long long N = readN(argc, argv);

    // ---------- 1) Создаём массив случайных чисел на rank = 0 ----------
    std::vector<double> data;  // полный массив будет только на rank 0
    if (rank == 0)
    {
        data.resize(N);

        // Генератор случайных чисел (равномерно в диапазоне [0, 1])
        std::mt19937_64 gen(12345); // фиксированный seed для повторяемости
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (long long i = 0; i < N; i++)
            data[i] = dist(gen);
    }

    // Засечём время (по методичке MPI_Wtime)
    double start_time = MPI_Wtime();

    // ---------- 2) Готовим параметры для MPI_Scatterv ----------
    // Scatterv позволяет раздать массив, даже если N не делится на size

    std::vector<int> sendcounts(size); // сколько элементов отправить каждому процессу
    std::vector<int> displs(size);     // смещения (с какого индекса отправлять)

    long long base = N / size;         // минимальный кусок
    long long rem  = N % size;         // остаток

    // Например: N=10, size=3 => base=3, rem=1
    // sendcounts: [4,3,3]
    long long offset = 0;
    for (int p = 0; p < size; p++)
    {
        long long cnt = base + (p < rem ? 1 : 0); // первым rem процессам даём +1
        sendcounts[p] = static_cast<int>(cnt);
        displs[p]     = static_cast<int>(offset);
        offset       += cnt;
    }

    // Локальный размер для текущего процесса
    int local_n = sendcounts[rank];

    // Локальный массив (часть данных для каждого процесса)
    std::vector<double> local(local_n);

    // Раздаём массив кусками
    MPI_Scatterv(
        rank == 0 ? data.data() : nullptr, // отправитель (только rank 0)
        sendcounts.data(),                 // сколько отправить каждому
        displs.data(),                     // смещения
        MPI_DOUBLE,                        // тип данных
        local.data(),                      // куда принимать
        local_n,                           // сколько принять
        MPI_DOUBLE,
        0,                                 // root = 0
        MPI_COMM_WORLD
    );

    // ---------- 3) Каждый процесс считает локальные суммы ----------
    // - сумма элементов
    // - сумма квадратов элементов
    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    for (int i = 0; i < local_n; i++)
    {
        local_sum += local[i];
        local_sum_sq += local[i] * local[i];
    }

    // ---------- 4) Собираем суммы на rank 0 через MPI_Reduce ----------
    double global_sum = 0.0;
    double global_sum_sq = 0.0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ---------- 5) На rank 0 считаем среднее и стандартное отклонение ----------
    if (rank == 0)
    {
        // Среднее: mean = sum / N
        double mean = global_sum / static_cast<double>(N);

        // Дисперсия: var = (sum(x^2)/N) - mean^2
        // Стандартное отклонение: std = sqrt(var)
        double ex2 = global_sum_sq / static_cast<double>(N);
        double var = ex2 - mean * mean;

        // Из-за погрешностей var может стать чуть отрицательной (например -1e-16)
        if (var < 0.0) var = 0.0;

        double stddev = std::sqrt(var);

        double end_time = MPI_Wtime();

        std::cout << "N = " << N << ", processes = " << size << "\n";
        std::cout << "Mean = " << mean << "\n";
        std::cout << "StdDev = " << stddev << "\n";
        std::cout << "Execution time: " << (end_time - start_time) << " seconds.\n";
    }

    MPI_Finalize();
    return 0;
}
