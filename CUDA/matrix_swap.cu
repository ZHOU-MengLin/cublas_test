#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                 \
                         cudaGetErrorString(status), __FILE__, __LINE__);     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status = (call);                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            std::fprintf(stderr, "cuBLAS error: %d (%s:%d)\n",               \
                         static_cast<int>(status), __FILE__, __LINE__);       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// 按矩阵的逻辑行列形式打印列主序数据
void print_matrix(const std::vector<double>& A,
                  int rows,
                  int cols,
                  int lda)
{
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            // 列主序索引：A[row + col * lda]
            std::printf("%6.1f ", A[row + col * lda]);
        }
        std::printf("\n");
    }
}

int main()
{
    const int rows = 3;
    const int cols = 4;
    const int lda = rows;

    /*
     * 逻辑矩阵：
     *
     *     1   2   3   4
     *     5   6   7   8
     *     9  10  11  12
     *
     * 列主序内存布局：
     *
     * [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
     */
    std::vector<double> A = {
         1.0,  5.0,  9.0,   // 第 0 列
         2.0,  6.0, 10.0,   // 第 1 列
         3.0,  7.0, 11.0,   // 第 2 列
         4.0,  8.0, 12.0    // 第 3 列
    };

    // 使用从 0 开始的行号
    const int row1 = 0;  // 第 1 行
    const int row2 = 2;  // 第 3 行

    std::printf("Before swapping row %d and row %d:\n",
                row1 + 1, row2 + 1);
    print_matrix(A, rows, cols, lda);

    double* d_A = nullptr;
    cublasHandle_t handle = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_A),
        A.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        A.data(),
        A.size() * sizeof(double),
        cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate(&handle));

    /*
     * 交换矩阵的两行：
     *
     * n    = cols：每行有 cols 个元素
     * x    = d_A + row1：第 row1 行的首个元素
     * incx = lda：移动到该行下一列的元素
     * y    = d_A + row2：第 row2 行的首个元素
     * incy = lda
     *
     * 实际交换：
     *
     * A[row1 + 0*lda] <-> A[row2 + 0*lda]
     * A[row1 + 1*lda] <-> A[row2 + 1*lda]
     * A[row1 + 2*lda] <-> A[row2 + 2*lda]
     * A[row1 + 3*lda] <-> A[row2 + 3*lda]
     */
    CUBLAS_CHECK(cublasDswap(
        handle,
        cols,
        d_A + row1,
        lda,
        d_A + row2,
        lda));

    CUDA_CHECK(cudaMemcpy(
        A.data(),
        d_A,
        A.size() * sizeof(double),
        cudaMemcpyDeviceToHost));

    // 捕获异步执行阶段可能产生的错误
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("\nAfter swapping row %d and row %d:\n",
                row1 + 1, row2 + 1);
    print_matrix(A, rows, cols, lda);

    const std::vector<double> expected = {
         9.0,  5.0,  1.0,
        10.0,  6.0,  2.0,
        11.0,  7.0,  3.0,
        12.0,  8.0,  4.0
    };

    const bool passed = A == expected;
    std::printf("\nResult: %s\n", passed ? "PASS" : "FAIL");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}