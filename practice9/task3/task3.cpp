#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

static const int INF = 1000000000; // "бесконечность" для отсутствия ребра

// Печать матрицы (только rank=0)
void print_matrix(const std::vector<int>& M, int N)
{
    std::cout << "Матрица кратчайших путей (dist):\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int v = M[i * N + j];
            if (v >= INF / 2) std::cout << std::setw(5) << "INF";
            else              std::cout << std::setw(5) << v;
        }
        std::cout << "\n";
    }
}

// Генерация матрицы смежности на rank=0
// - диагональ = 0
// - часть рёбер отсутствует (INF)
// - веса рёбер 1..20
void generate_graph(std::vector<int>& G, int N)
{
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> wdist(1, 20);
    std::uniform_real_distribution<double> pdist(0.0, 1.0);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i == j) { G[i * N + j] = 0; continue; }

            // вероятность существования ребра ~ 70%
            if (pdist(gen) < 0.7) G[i * N + j] = wdist(gen);
            else                  G[i * N + j] = INF;
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер графа N берём из аргументов или по умолчанию
    int N = 8;
    if (argc >= 2) N = std::atoi(argv[1]);
    if (N <= 0)
    {
        if (rank == 0) std::cerr << "Ошибка: N должен быть > 0\n";
        MPI_Finalize();
        return 1;
    }

    // По заданию используем MPI_Scatter -> нужно равное число строк на процесс
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

    const int rows_per_proc = N / size;

    // ---------- 1) rank=0 создаёт матрицу смежности ----------
    std::vector<int> G; // полная матрица только на rank=0
    if (rank == 0)
    {
        G.resize(N * N);
        generate_graph(G, N);
    }

    // ---------- 2) Раздаём строки матрицы по процессам ----------
    // local: rows_per_proc строк, каждая по N элементов
    std::vector<int> local(rows_per_proc * N);

    MPI_Scatter(
        rank == 0 ? G.data() : nullptr,
        rows_per_proc * N,
        MPI_INT,
        local.data(),
        rows_per_proc * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Для алгоритма Флойда–Уоршелла нам нужна "глобальная" матрица dist,
    // чтобы брать строку k на каждой итерации.
    // Мы будем восстанавливать глобальную матрицу через MPI_Allgather
    // после каждого шага k.
    std::vector<int> dist_global(N * N);

    // Сначала соберём исходное состояние (после Scatter) в dist_global
    MPI_Allgather(
        local.data(),
        rows_per_proc * N,
        MPI_INT,
        dist_global.data(),
        rows_per_proc * N,
        MPI_INT,
        MPI_COMM_WORLD
    );

    double start_time = MPI_Wtime();

    // Буфер для строки k (нужна всем процессам)
    std::vector<int> row_k(N);

    // ---------- 3) Алгоритм Флойда–Уоршелла ----------
    // dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    for (int k = 0; k < N; k++)
    {
        // Забираем строку k из глобальной матрицы (она у всех одинаковая)
        for (int j = 0; j < N; j++)
            row_k[j] = dist_global[k * N + j];

        // Каждый процесс обновляет только свои строки (локальные)
        // Глобальный индекс строки i_global = rank*rows_per_proc + i_local
        int base_row = rank * rows_per_proc;

        for (int i_local = 0; i_local < rows_per_proc; i_local++)
        {
            int i_global = base_row + i_local;

            // dist[i][k] берём из dist_global, потому что k может быть в чужом блоке
            int dik = dist_global[i_global * N + k];
            if (dik >= INF / 2) continue; // если пути i->k нет, обновлять бессмысленно

            for (int j = 0; j < N; j++)
            {
                int dkj = row_k[j];
                if (dkj >= INF / 2) continue; // если k->j нет

                int candidate = dik + dkj;

                // local хранит нашу строку i_local, столбец j
                int& dij = local[i_local * N + j];
                if (candidate < dij) dij = candidate;
            }
        }

        // После обновления локальных строк нужно, чтобы ВСЕ процессы
        // увидели новую глобальную матрицу для следующего k.
        MPI_Allgather(
            local.data(),
            rows_per_proc * N,
            MPI_INT,
            dist_global.data(),
            rows_per_proc * N,
            MPI_INT,
            MPI_COMM_WORLD
        );
    }

    double end_time = MPI_Wtime();

    // ---------- 4) Вывод результата (только rank=0) ----------
    if (rank == 0)
    {
        std::cout << "Практическая №9 / Таск 3: Флойд–Уоршелл (MPI)\n";
        std::cout << "N = " << N << ", процессов = " << size << "\n";
        std::cout << "Время выполнения: " << (end_time - start_time) << " сек\n\n";

        print_matrix(dist_global, N);
    }

    MPI_Finalize();
    return 0;
}
