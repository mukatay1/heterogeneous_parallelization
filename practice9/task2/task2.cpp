#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cstdlib>

// Печать решения (только root)
static void print_solution(const std::vector<double>& x)
{
    std::cout << "Решение x:\n";
    for (size_t i = 0; i < x.size(); i++)
        std::cout << "x[" << i << "] = " << std::setprecision(10) << x[i] << "\n";
}

// Быстрая генерация системы A и b (на root)
static void generate_system(std::vector<double>& Ab, int N)
{
    // Ab хранит расширенную матрицу [A|b] размером N x (N+1)
    // Заполним A так, чтобы система была хорошо обусловлена:
    // диагональ сделаем "побольше", чтобы не было нулевых опорных элементов.
    std::mt19937_64 gen(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < N; i++)
    {
        double rowsum = 0.0;
        for (int j = 0; j < N; j++)
        {
            double v = dist(gen);
            Ab[i * (N + 1) + j] = v;
            rowsum += std::fabs(v);
        }
        // Усиливаем диагональ (чтобы pivot почти точно не был 0)
        Ab[i * (N + 1) + i] += rowsum + 1.0;

        // Правая часть b
        Ab[i * (N + 1) + N] = dist(gen);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер системы N берём из аргументов или по умолчанию
    int N = 8;
    if (argc >= 2) N = std::atoi(argv[1]);
    if (N <= 0)
    {
        if (rank == 0) std::cerr << "Ошибка: N должен быть > 0\n";
        MPI_Finalize();
        return 1;
    }

    // Так как в задании указан MPI_Scatter, требуем N % size == 0
    if (N % size != 0)
    {
        if (rank == 0)
        {
            std::cerr << "Ошибка: для MPI_Scatter нужно, чтобы N делилось на число процессов.\n";
            std::cerr << "Сейчас N=" << N << ", процессов=" << size << " => N % P = " << (N % size) << "\n";
            std::cerr << "Выберите N кратное P или используйте Scatterv (но в этом таске требуется Scatter).\n";
        }
        MPI_Finalize();
        return 2;
    }

    const int rows_per_proc = N / size;     // сколько строк у каждого процесса
    const int cols_aug = N + 1;             // ширина расширенной матрицы [A|b]

    // ---------------- 1) Root создаёт систему [A|b] ----------------
    std::vector<double> Ab;                 // полная матрица только на root
    if (rank == 0)
    {
        Ab.resize(N * cols_aug);
        generate_system(Ab, N);
    }

    // ---------------- 2) Каждый процесс получает свои строки ----------------
    // localAb: rows_per_proc строк, каждая длиной (N+1)
    std::vector<double> localAb(rows_per_proc * cols_aug);

    // Scatter раздаёт равные блоки: каждому по rows_per_proc строк
    MPI_Scatter(
        rank == 0 ? Ab.data() : nullptr,            // откуда раздаём (root)
        rows_per_proc * cols_aug,                   // сколько элементов отправить каждому
        MPI_DOUBLE,
        localAb.data(),                              // куда принимаем (у каждого)
        rows_per_proc * cols_aug,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Для замера времени
    double start_time = MPI_Wtime();

    // Буфер для опорной строки (pivot row), которую будем Bcast-ить всем
    std::vector<double> pivot_row(cols_aug);

    // ---------------- 3) Прямой ход метода Гаусса ----------------
    // Идём по столбцам k = 0..N-1 и зануляем элементы ниже диагонали
    for (int k = 0; k < N; k++)
    {
        // Определяем, какой процесс владеет глобальной строкой k
        int owner = k / rows_per_proc;

        // Если текущий процесс владеет строкой k — копируем её в pivot_row
        if (rank == owner)
        {
            int local_k = k - rank * rows_per_proc; // локальный индекс строки k у владельца
            for (int j = 0; j < cols_aug; j++)
                pivot_row[j] = localAb[local_k * cols_aug + j];
        }

        // Рассылаем pivot_row всем процессам
        MPI_Bcast(pivot_row.data(), cols_aug, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Проверка на нулевой pivot (без частичного выбора главного элемента)
        double pivot = pivot_row[k];
        if (std::fabs(pivot) < 1e-12)
        {
            if (rank == 0)
                std::cerr << "Предупреждение: найден почти нулевой pivot на k=" << k
                          << ". Без pivoting решение может быть неверным.\n";
            // Продолжаем, но это риск
        }

        // Каждый процесс обрабатывает ТОЛЬКО свои строки ниже k
        int global_start = rank * rows_per_proc;

        for (int i_local = 0; i_local < rows_per_proc; i_local++)
        {
            int i_global = global_start + i_local;

            if (i_global <= k) continue; // только строки ниже диагонали

            double* row = &localAb[i_local * cols_aug];

            double factor = 0.0;
            if (std::fabs(pivot) > 1e-12)
                factor = row[k] / pivot;

            // Вычитаем factor * pivot_row из текущей строки
            // Начинаем с j=k (можно с k, чтобы экономить операции)
            for (int j = k; j < cols_aug; j++)
                row[j] -= factor * pivot_row[j];

            // Явно зануляем (для стабильности вывода)
            row[k] = 0.0;
        }
    }

    // ---------------- 4) Собираем верхнетреугольную матрицу на root ----------------
    MPI_Gather(
        localAb.data(),
        rows_per_proc * cols_aug,
        MPI_DOUBLE,
        rank == 0 ? Ab.data() : nullptr,
        rows_per_proc * cols_aug,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // ---------------- 5) Обратный ход (делаем на root) ----------------
    if (rank == 0)
    {
        std::vector<double> x(N, 0.0);

        // Back substitution
        for (int i = N - 1; i >= 0; i--)
        {
            double sum = 0.0;
            for (int j = i + 1; j < N; j++)
                sum += Ab[i * cols_aug + j] * x[j];

            double diag = Ab[i * cols_aug + i];
            double rhs  = Ab[i * cols_aug + N];

            if (std::fabs(diag) < 1e-12)
            {
                std::cerr << "Ошибка: диагональный элемент близок к 0 в строке " << i
                          << ". Решение может не существовать/быть неединственным.\n";
                x[i] = 0.0; // условно
            }
            else
            {
                x[i] = (rhs - sum) / diag;
            }
        }

        double end_time = MPI_Wtime();

        std::cout << "Практическая №9 / Таск 2: Метод Гаусса (MPI)\n";
        std::cout << "N = " << N << ", процессов = " << size << "\n";
        std::cout << "Время выполнения: " << (end_time - start_time) << " сек\n\n";

        print_solution(x);
    }

    MPI_Finalize();
    return 0;
}
