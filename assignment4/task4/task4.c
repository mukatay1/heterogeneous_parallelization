#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Функция инициализации массива начальными значениями
// Используется только корневым процессом
static void fill_array(double* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (double)i * 0.001;
    }
}

int main(int argc, char** argv) {

    // Инициализация MPI среды
    MPI_Init(&argc, &argv);

    // rank — номер текущего процесса
    // size — общее количество процессов
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер обрабатываемого массива
    // Можно передать через аргументы командной строки
    long long N = 10000000; // 10 млн элементов по умолчанию
    if (argc >= 2) {
        N = atoll(argv[1]);
    }

    // Коэффициент для вычислений
    const double alpha = 3.14159;

    // sendcounts[p] — сколько элементов отправляется процессу p
    // displs[p]     — смещение (индекс) начала блока для процесса p
    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs     = (int*)malloc(size * sizeof(int));

    // Базовый размер блока для каждого процесса
    long long base = N / size;

    // Остаток, который распределяется между первыми процессами
    long long remainder = N % size;

    // Вычисление размеров и смещений
    long long offset = 0;
    for (int p = 0; p < size; p++) {
        long long chunk = base + (p < remainder ? 1 : 0);
        sendcounts[p] = (int)chunk;
        displs[p]     = (int)offset;
        offset += chunk;
    }

    // Количество элементов, обрабатываемых текущим процессом
    int local_n = sendcounts[rank];

    // Локальные массивы для приёма данных и хранения результата
    double* local_x = (double*)malloc(local_n * sizeof(double));
    double* local_y = (double*)malloc(local_n * sizeof(double));

    // Глобальные массивы существуют только у корневого процесса
    double* x = NULL;
    double* y = NULL;

    if (rank == 0) {
        // Выделение памяти под исходный и результирующий массивы
        x = (double*)malloc(N * sizeof(double));
        y = (double*)malloc(N * sizeof(double));

        // Инициализация исходных данных
        fill_array(x, (int)N);
    }

    // Барьер синхронизации перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Каждый процесс получает свою часть массива x
    MPI_Scatterv(
        x,                 // исходный массив (только у rank 0)
        sendcounts,         // размеры блоков
        displs,             // смещения
        MPI_DOUBLE,         // тип данных
        local_x,            // локальный буфер
        local_n,            // количество элементов
        MPI_DOUBLE,
        0,                  // корневой процесс
        MPI_COMM_WORLD
    );

    // Каждый процесс независимо обрабатывает свой участок данных
    for (int i = 0; i < local_n; i++) {
        local_y[i] = local_x[i] * alpha;
    }

    // Результаты локальных вычислений собираются в массив y
    MPI_Gatherv(
        local_y,            // локальный результат
        local_n,
        MPI_DOUBLE,
        y,                  // итоговый массив (rank 0)
        sendcounts,
        displs,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Барьер для корректного завершения измерений
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Локальное время выполнения
    double local_time = end_time - start_time;

    // Находим максимальное время среди всех процессов
    // Именно оно определяет реальное время параллельного выполнения
    double max_time = 0.0;
    MPI_Reduce(
        &local_time,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    // Вывод результатов только корневым процессом
    if (rank == 0) {
        printf("Array size: %lld\n", N);
        printf("Processes: %d\n", size);
        printf("Execution time: %.6f seconds\n", max_time);

        // Небольшая проверка корректности вычислений
        int test_indices[3] = {0, (int)(N / 2), (int)(N - 1)};
        for (int k = 0; k < 3; k++) {
            int i = test_indices[k];
            double expected = x[i] * alpha;
            double diff = y[i] - expected;
            if (diff < 0) diff = -diff;
            printf("Check y[%d]: diff = %.6e\n", i, diff);
        }
    }

    // Освобождение памяти
    free(local_x);
    free(local_y);
    free(sendcounts);
    free(displs);
    if (rank == 0) {
        free(x);
        free(y);
    }

    // Завершение работы MPI
    MPI_Finalize();
    return 0;
}
