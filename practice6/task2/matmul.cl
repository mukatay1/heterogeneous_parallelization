// Умножение матриц: C = A * B
// A: N x M
// B: M x K
// C: N x K
__kernel void matmul(__global const float* A,
                     __global const float* B,
                     __global float* C,
                     const int N,
                     const int M,
                     const int K)
{
    // row — индекс строки в C (и в A)
    // col — индекс столбца в C (и в B)
    int row = get_global_id(0);
    int col = get_global_id(1);

    // Защита на случай, если глобальные размеры округлялись вверх
    if (row >= N || col >= K) return;

    float sum = 0.0f;

    // Скалярное произведение строки A и столбца B
    for (int i = 0; i < M; i++) {
        // A[row][i] = A[row*M + i]
        // B[i][col] = B[i*K + col]
        sum += A[row * M + i] * B[i * K + col];
    }

    // C[row][col] = sum
    C[row * K + col] = sum;
}
