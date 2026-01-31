#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cstdlib>

// ------------------ Генерация данных ------------------
// Создаём массив случайных чисел double в диапазоне [0,1)
static void fill_random(std::vector<double>& a)
{
    std::mt19937_64 gen(12345); // фиксированный seed для повторяемости
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto &x : a) x = dist(gen);
}

// ------------------ Последовательная версия ------------------
// Считаем сумму, среднее и дисперсию (двухпроходный вариант: устойчивее)
static void stats_serial(const std::vector<double>& a, double& sum, double& mean, double& var)
{
    const long long n = (long long)a.size();
    sum = 0.0;

    for (long long i = 0; i < n; i++)
        sum += a[i];

    mean = sum / (double)n;

    double acc = 0.0;
    for (long long i = 0; i < n; i++)
    {
        double d = a[i] - mean;
        acc += d * d;
    }

    var = acc / (double)n;
}

// ------------------ Параллельная версия (OpenMP) ------------------
// 1) Сумма и среднее через reduction
// 2) Дисперсия вторым проходом тоже через reduction
static void stats_openmp(const std::vector<double>& a, double& sum, double& mean, double& var, int threads)
{
    const long long n = (long long)a.size();
    sum = 0.0;

    // Указываем число потоков для данного участка
    omp_set_num_threads(threads);

    // --- Параллельный подсчёт суммы ---
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++)
        sum += a[i];

    mean = sum / (double)n;

    // --- Параллельный подсчёт суммы квадратов отклонений ---
    double acc = 0.0;
    #pragma omp parallel for reduction(+:acc)
    for (long long i = 0; i < n; i++)
    {
        double d = a[i] - mean;
        acc += d * d;
    }

    var = acc / (double)n;
}

// ------------------ Сравнение double с допуском ------------------
static bool close_enough(double x, double y, double eps = 1e-9)
{
    return std::fabs(x - y) <= eps * (1.0 + std::fabs(x) + std::fabs(y));
}

int main(int argc, char** argv)
{
    // Размер массива
    long long N = 10000000LL; // 10 млн по умолчанию
    if (argc >= 2) N = std::atoll(argv[1]);
    if (N <= 0)
    {
        std::cerr << "Ошибка: N должен быть > 0\n";
        return 1;
    }

    std::cout << "Практическая №10 / Таск 1: OpenMP профилирование CPU\n";
    std::cout << "N = " << N << "\n\n";

    // 1) Создаём данные
    std::vector<double> a((size_t)N);

    double t_gen0 = omp_get_wtime();
    fill_random(a);
    double t_gen1 = omp_get_wtime();
    double t_gen = t_gen1 - t_gen0;

    // 2) Последовательный базовый прогон (1 поток, без OpenMP)
    double s_sum = 0.0, s_mean = 0.0, s_var = 0.0;
    double t_ser0 = omp_get_wtime();
    stats_serial(a, s_sum, s_mean, s_var);
    double t_ser1 = omp_get_wtime();
    double t_serial = t_ser1 - t_ser0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Время генерации данных: " << t_gen << " сек\n";
    std::cout << "Последовательное время (baseline): " << t_serial << " сек\n\n";

    // 3) Параллельные прогоны для разных чисел потоков
    int max_threads = omp_get_max_threads();

    // Список потоков для теста (1,2,4,8,...,max_threads)
    std::vector<int> thread_list;
    for (int t = 1; t <= max_threads; t *= 2) thread_list.push_back(t);
    if (thread_list.back() != max_threads) thread_list.push_back(max_threads);

    // Для оценки закона Амдала нам нужна доля параллельной части P (или доля последовательной S)
    // Мы будем оценивать S по формуле Амдала из измеренного ускорения:
    // Speedup(T) = 1 / ( S + (1-S)/T )  =>  S = (1/Speedup - 1/T) / (1 - 1/T)
    // Возьмём оценку S по самому большому T (обычно стабильнее).

    double best_S_est = -1.0;

    std::cout << "Тест OpenMP:\n";
    std::cout << "threads | time(s) | speedup | efficiency\n";
    std::cout << "----------------------------------------\n";

    for (int threads : thread_list)
    {
        double p_sum = 0.0, p_mean = 0.0, p_var = 0.0;

        // Меряем только вычисление статистик
        double t0 = omp_get_wtime();
        stats_openmp(a, p_sum, p_mean, p_var, threads);
        double t1 = omp_get_wtime();
        double t_par = t1 - t0;

        // Проверка корректности относительно последовательной версии
        bool ok =
            close_enough(p_sum,  s_sum)  &&
            close_enough(p_mean, s_mean) &&
            close_enough(p_var,  s_var);

        double speedup = t_serial / t_par;
        double eff = speedup / (double)threads;

        std::cout << std::setw(7) << threads
                  << " | " << std::setw(7) << t_par
                  << " | " << std::setw(7) << speedup
                  << " | " << std::setw(10) << eff
                  << (ok ? "" : "   <-- ВНИМАНИЕ: расхождение!") << "\n";

        // Оценка доли последовательной части S по Амдалу (только если threads > 1)
        if (threads > 1)
        {
            double invSp = 1.0 / speedup;
            double invT  = 1.0 / (double)threads;
            double denom = (1.0 - invT);

            // S = (1/Sp - 1/T) / (1 - 1/T)
            double S = (invSp - invT) / denom;

            // Берём оценку по максимальному числу потоков
            if (threads == thread_list.back())
                best_S_est = S;
        }
    }

    std::cout << "\n";

    // 4) Анализ по закону Амдала
    if (best_S_est > 0.0)
    {
        double S = best_S_est;        // доля последовательной части
        double P = 1.0 - S;           // доля параллельной части

        // Теоретический максимум ускорения при T -> бесконечности: 1/S
        double max_speedup = 1.0 / S;

        std::cout << "Анализ (закон Амдала):\n";
        std::cout << "Оценка доли последовательной части S ≈ " << S << "\n";
        std::cout << "Оценка доли параллельной части   P ≈ " << P << "\n";
        std::cout << "Теоретический предел ускорения (T→∞): 1/S ≈ " << max_speedup << "x\n";
        std::cout << "\n";
        std::cout << "Комментарий:\n";
        std::cout << "- Если S заметно больше 0, ускорение будет быстро 'упираться' в потолок.\n";
        std::cout << "- Если эффективность сильно падает при росте потоков, возможны причины:\n";
        std::cout << "  конкуренция за память/кэш, NUMA, накладные расходы OpenMP, малая работа на поток.\n";
    }
    else
    {
        std::cout << "Не удалось оценить S по Амдалу (попробуй запустить с >=2 потоками).\n";
    }

    return 0;
}
