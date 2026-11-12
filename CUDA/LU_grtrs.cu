#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(err) << std::endl;       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = (call);                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__   \
                      << ": status = " << static_cast<int>(status)           \
                      << std::endl;                                           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// nvcc LU_grtrs.cu -o LU_grtrs -lcublas
// nvcc -O2 -std=c++17 -o LU_grtrs LU_grtrs.cu -lcublas
int main()
{
    // ============================================================
    // Problem configuration
    // ============================================================

    constexpr int N = 3;
    constexpr int NRHS = 1;
    constexpr int BatchCount = 2;

    const size_t MatrixSize =
        static_cast<size_t>(N) * N;

    const size_t BSize =
        static_cast<size_t>(N) * NRHS;


    // ============================================================
    // Host data
    // ============================================================

    std::vector<std::vector<float>> h_A(BatchCount);
    std::vector<std::vector<float>> h_B(BatchCount);

    // Keep original A/B for residual checking.
    std::vector<std::vector<float>> h_A_original(BatchCount);
    std::vector<std::vector<float>> h_B_original(BatchCount);

    for (int batch = 0; batch < BatchCount; ++batch) {
        h_A[batch].resize(MatrixSize);
        h_B[batch].resize(BSize);

        h_A_original[batch].resize(MatrixSize);
        h_B_original[batch].resize(BSize);
    }


    // ============================================================
    // Batch 0
    //
    // A0 =
    // [ 3  1  2
    //   1  4  1
    //   2  1  5 ]
    //
    // B0 =
    // [ 1
    //   2
    //   3 ]
    //
    // Column-major storage.
    // ============================================================

    h_A[0] = {
        3.0f, 5.0f, 6.0f,
        1.0f, 4.0f, 1.0f,
        2.0f, 1.0f, 5.0f
    };

    h_B[0] = {
        5.0f,
        2.0f,
        3.0f
    };


    // ============================================================
    // Batch 1
    //
    // A1 =
    // [ 4  1  0
    //   1  3  1
    //   0  1  2 ]
    //
    // B1 =
    // [ 1
    //   2
    //   3 ]
    // ============================================================

    h_A[1] = {
        4.0f, 1.0f, 0.0f,
        1.0f, 3.0f, 1.0f,
        0.0f, 1.0f, 2.0f
    };

    h_B[1] = {
        1.0f,
        2.0f,
        3.0f
    };


    // Save original matrices.
    for (int batch = 0; batch < BatchCount; ++batch) {
        h_A_original[batch] = h_A[batch];
        h_B_original[batch] = h_B[batch];
    }


    // ============================================================
    // Device matrices
    // ============================================================

    std::vector<float*> d_A(BatchCount, nullptr);
    std::vector<float*> d_B(BatchCount, nullptr);

    for (int batch = 0; batch < BatchCount; ++batch) {

        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_A[batch]),
            MatrixSize * sizeof(float)));

        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_B[batch]),
            BSize * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(
            d_A[batch],
            h_A[batch].data(),
            MatrixSize * sizeof(float),
            cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemcpy(
            d_B[batch],
            h_B[batch].data(),
            BSize * sizeof(float),
            cudaMemcpyHostToDevice));
    }


    // ============================================================
    // Device pointer arrays
    // ============================================================

    float** d_Aarray = nullptr;
    float** d_Barray = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_Aarray),
        BatchCount * sizeof(float*)));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_Barray),
        BatchCount * sizeof(float*)));

    CHECK_CUDA(cudaMemcpy(
        d_Aarray,
        d_A.data(),
        BatchCount * sizeof(float*),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_Barray,
        d_B.data(),
        BatchCount * sizeof(float*),
        cudaMemcpyHostToDevice));


    // ============================================================
    // cuBLAS handle
    // ============================================================

    cublasHandle_t handle;

    CHECK_CUBLAS(
        cublasCreate(&handle));


    // ============================================================
    // Pivot array
    //
    // DEVICE memory
    //
    // Size = N * BatchCount
    // ============================================================

    int* d_PivotArray = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_PivotArray),
        BatchCount * N * sizeof(int)));


    // ============================================================
    // getrf info array
    //
    // IMPORTANT:
    //
    // cublasSgetrfBatched:
    //     infoArray is DEVICE memory.
    // ============================================================

    int* d_InfoArray = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_InfoArray),
        BatchCount * sizeof(int)));

    // CHECK_CUDA(cudaMemset(
    //     d_InfoArray,
    //     0,
    //     BatchCount * sizeof(int)));


    // ============================================================
    // Step 1: LU factorization
    //
    // A -> LU
    //
    // In-place.
    // ============================================================

    std::cout << "Calling cublasSgetrfBatched..." << std::endl;

    CHECK_CUBLAS(
        cublasSgetrfBatched(
            handle,
            N,
            d_Aarray,
            N,
            d_PivotArray,
            d_InfoArray,
            BatchCount));

    CHECK_CUDA(
        cudaDeviceSynchronize());

    std::cout << "cublasSgetrfBatched finished." << std::endl;


    // ============================================================
    // Copy getrf info back to host
    // ============================================================

    std::vector<int> h_InfoArray(BatchCount);

    CHECK_CUDA(cudaMemcpy(
        h_InfoArray.data(),
        d_InfoArray,
        BatchCount * sizeof(int),
        cudaMemcpyDeviceToHost));


    for (int batch = 0; batch < BatchCount; ++batch) {

        std::cout
            << "getrf batch "
            << batch
            << " info = "
            << h_InfoArray[batch]
            << std::endl;

        if (h_InfoArray[batch] != 0) {
            std::cerr
                << "LU factorization failed for batch "
                << batch
                << std::endl;

            return EXIT_FAILURE;
        }
    }


    // ============================================================
    // Step 2: Solve
    //
    // IMPORTANT:
    //
    // cublasSgetrsBatched:
    //
    //     info is HOST memory.
    //
    // Do NOT pass d_InfoArray here.
    // ============================================================

    int h_Info = 0;

    std::cout << "Calling cublasSgetrsBatched..." << std::endl;

    CHECK_CUBLAS(
        cublasSgetrsBatched(
            handle,
            CUBLAS_OP_N,
            N,
            NRHS,

            reinterpret_cast<
                const float* const*
            >(d_Aarray),

            N,

            d_PivotArray,

            d_Barray,

            N,

            &h_Info,

            BatchCount));

    CHECK_CUDA(
        cudaDeviceSynchronize());

    std::cout << "cublasSgetrsBatched finished." << std::endl;

    std::cout
        << "getrs info = "
        << h_Info
        << std::endl;


    if (h_Info != 0) {
        std::cerr
            << "SgetrsBatched failed, info = "
            << h_Info
            << std::endl;

        return EXIT_FAILURE;
    }


    // ============================================================
    // Copy solution X back
    // ============================================================

    std::vector<std::vector<float>> h_X(BatchCount);

    for (int batch = 0; batch < BatchCount; ++batch) {

        h_X[batch].resize(BSize);

        CHECK_CUDA(cudaMemcpy(
            h_X[batch].data(),
            d_B[batch],
            BSize * sizeof(float),
            cudaMemcpyDeviceToHost));
    }


    // ============================================================
    // Print solution
    // ============================================================

    std::cout
        << std::fixed
        << std::setprecision(6);

    for (int batch = 0; batch < BatchCount; ++batch) {

        std::cout
            << "\nBatch "
            << batch
            << " solution X:"
            << std::endl;

        for (int row = 0; row < N; ++row) {

            std::cout << "  ";

            for (int col = 0; col < NRHS; ++col) {

                std::cout
                    << std::setw(12)
                    << h_X[batch][row + col * N];
            }

            std::cout << std::endl;
        }
    }


    // ============================================================
    // Residual check
    //
    // R = A_original * X - B_original
    // ============================================================

    std::cout << "\nResidual check:" << std::endl;

    for (int batch = 0; batch < BatchCount; ++batch) {

        float max_error = 0.0f;

        for (int rhs = 0; rhs < NRHS; ++rhs) {

            for (int row = 0; row < N; ++row) {

                float value = 0.0f;

                for (int col = 0; col < N; ++col) {

                    value +=
                        h_A_original[batch][row + col * N] *
                        h_X[batch][col + rhs * N];
                }

                value -=
                    h_B_original[batch][row + rhs * N];

                max_error =
                    std::max(
                        max_error,
                        std::fabs(value));
            }
        }

        std::cout
            << "Batch "
            << batch
            << " max |A*X-B| = "
            << max_error
            << std::endl;
    }


    // ============================================================
    // Cleanup
    // ============================================================

    CHECK_CUBLAS(
        cublasDestroy(handle));

    CHECK_CUDA(
        cudaFree(d_PivotArray));

    CHECK_CUDA(
        cudaFree(d_InfoArray));

    CHECK_CUDA(
        cudaFree(d_Aarray));

    CHECK_CUDA(
        cudaFree(d_Barray));

    for (int batch = 0; batch < BatchCount; ++batch) {

        CHECK_CUDA(
            cudaFree(d_A[batch]));

        CHECK_CUDA(
            cudaFree(d_B[batch]));
    }

    return EXIT_SUCCESS;
}