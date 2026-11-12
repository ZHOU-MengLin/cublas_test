#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

//nvcc getrs_largeN.cu -o getrs_largeN -lcublas

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                   \
    } while (0)

#define CHECK_CUBLAS(call)                                                  \
    do {                                                                    \
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "cuBLAS error: "                                 \
                      << static_cast<int>(status)                           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                   \
    } while (0)


static constexpr int kWarmup = 5;
static constexpr int kRepeat = 20;


struct TestConfig {
    int n;
    int nrhs;
    int batch_size;
};


void FillRandom(std::vector<float>& data) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& value : data) {
        value = dist(gen);
    }
}


bool RunTest(
    cublasHandle_t handle,
    const TestConfig& config) {

    const int n = config.n;
    const int nrhs = config.nrhs;
    const int batch_size = config.batch_size;

    const size_t matrix_elements =
        static_cast<size_t>(n) * n;

    const size_t rhs_elements =
        static_cast<size_t>(n) * nrhs;

    const size_t total_a =
        matrix_elements * batch_size;

    const size_t total_b =
        rhs_elements * batch_size;

    std::cout << "\n========================================\n";
    std::cout << "N=" << n
              << ", nrhs=" << nrhs
              << ", batchSize=" << batch_size
              << "\n";

    std::cout << "A memory = "
              << static_cast<double>(
                     total_a * sizeof(float))
              / (1024.0 * 1024.0)
              << " MB\n";

    std::cout << "B memory = "
              << static_cast<double>(
                     total_b * sizeof(float))
              / (1024.0 * 1024.0)
              << " MB\n";


    /*
     * ============================================================
     * Host data
     * ============================================================
     */

    std::vector<float> h_A(total_a);
    std::vector<float> h_B(total_b);

    FillRandom(h_A);
    FillRandom(h_B);


    /*
     * A = random + N * I
     *
     * Make A diagonally dominant so that it is very unlikely
     * to be singular.
     */
    for (int batch = 0; batch < batch_size; ++batch) {

        float* A =
            h_A.data()
            + static_cast<size_t>(batch) * matrix_elements;

        for (int i = 0; i < n; ++i) {

            A[i + static_cast<size_t>(i) * n]
                += static_cast<float>(n);
        }
    }


    /*
     * ============================================================
     * Device memory
     * ============================================================
     */

    float* d_A = nullptr;
    float* d_B = nullptr;

    int* d_pivot = nullptr;
    int* d_info = nullptr;

    float** d_A_ptr = nullptr;
    const float** d_A_const_ptr = nullptr;
    float** d_B_ptr = nullptr;


    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_A),
        total_a * sizeof(float)));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_B),
        total_b * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(
        d_A,
        h_A.data(),
        total_a * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_B,
        h_B.data(),
        total_b * sizeof(float),
        cudaMemcpyHostToDevice));


    /*
     * ============================================================
     * A pointer array for Getrf
     * ============================================================
     */

    std::vector<float*> h_A_ptr(batch_size);

    for (int i = 0; i < batch_size; ++i) {

        h_A_ptr[i] =
            d_A
            + static_cast<size_t>(i) * matrix_elements;
    }

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_A_ptr),
        batch_size * sizeof(float*)));

    CHECK_CUDA(cudaMemcpy(
        d_A_ptr,
        h_A_ptr.data(),
        batch_size * sizeof(float*),
        cudaMemcpyHostToDevice));


    /*
     * ============================================================
     * A pointer array for Getrs
     *
     * IMPORTANT:
     *
     * Do not cast float** to const float**.
     * ============================================================
     */

    std::vector<const float*> h_A_const_ptr(batch_size);

    for (int i = 0; i < batch_size; ++i) {

        h_A_const_ptr[i] =
            d_A
            + static_cast<size_t>(i) * matrix_elements;
    }

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_A_const_ptr),
        batch_size * sizeof(const float*)));

    CHECK_CUDA(cudaMemcpy(
        d_A_const_ptr,
        h_A_const_ptr.data(),
        batch_size * sizeof(const float*),
        cudaMemcpyHostToDevice));


    /*
     * ============================================================
     * B pointer array
     * ============================================================
     */

    std::vector<float*> h_B_ptr(batch_size);

    for (int i = 0; i < batch_size; ++i) {

        h_B_ptr[i] =
            d_B
            + static_cast<size_t>(i) * rhs_elements;
    }

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_B_ptr),
        batch_size * sizeof(float*)));

    CHECK_CUDA(cudaMemcpy(
        d_B_ptr,
        h_B_ptr.data(),
        batch_size * sizeof(float*),
        cudaMemcpyHostToDevice));


    /*
     * ============================================================
     * Pivot
     * ============================================================
     */

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_pivot),
        static_cast<size_t>(n)
            * batch_size
            * sizeof(int)));


    /*
     * ============================================================
     * Info
     * ============================================================
     */

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_info),
        batch_size * sizeof(int)));


    /*
     * ============================================================
     * GETRF
     * ============================================================
     */

    std::cout << "Running Getrf..." << std::endl;

    cublasStatus_t status =
        cublasSgetrfBatched(
            handle,
            n,
            d_A_ptr,
            n,
            d_pivot,
            d_info,
            batch_size);

    if (status != CUBLAS_STATUS_SUCCESS) {

        std::cerr
            << "cublasSgetrfBatched failed, status="
            << static_cast<int>(status)
            << "\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_A_ptr);
        cudaFree(d_A_const_ptr);
        cudaFree(d_B_ptr);
        cudaFree(d_pivot);
        cudaFree(d_info);

        return false;
    }

    CHECK_CUDA(cudaDeviceSynchronize());


    /*
     * Check Getrf info
     */

    std::vector<int> h_info(batch_size);

    CHECK_CUDA(cudaMemcpy(
        h_info.data(),
        d_info,
        batch_size * sizeof(int),
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch_size; ++i) {

        if (h_info[i] != 0) {

            std::cerr
                << "Getrf failed: batch="
                << i
                << ", info="
                << h_info[i]
                << "\n";

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_A_ptr);
            cudaFree(d_A_const_ptr);
            cudaFree(d_B_ptr);
            cudaFree(d_pivot);
            cudaFree(d_info);

            return false;
        }
    }

    std::cout << "Getrf PASS\n";


    /*
     * ============================================================
     * GETRS
     * ============================================================
     */

    int h_Info = 0;
    std::cout << "Running Getrs warmup..." << std::endl;

    for (int i = 0; i < kWarmup; ++i) {

        status =
            cublasSgetrsBatched(
                handle,
                CUBLAS_OP_N,
                n,
                nrhs,
                d_A_const_ptr,
                n,
                d_pivot,
                d_B_ptr,
                n,
                &h_Info,
                batch_size);

        if (status != CUBLAS_STATUS_SUCCESS) {

            std::cerr
                << "cublasSgetrsBatched failed during warmup, "
                << "status="
                << static_cast<int>(status)
                << "\n";

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_A_ptr);
            cudaFree(d_A_const_ptr);
            cudaFree(d_B_ptr);
            cudaFree(d_pivot);

            return false;
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Getrs warmup PASS\n";


    /*
     * ============================================================
     * Benchmark
     * ============================================================
     */

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < kRepeat; ++i) {

        status =
            cublasSgetrsBatched(
                handle,
                CUBLAS_OP_N,
                n,
                nrhs,
                d_A_const_ptr,
                n,
                d_pivot,
                d_B_ptr,
                n,
                &h_Info,
                batch_size);

        if (status != CUBLAS_STATUS_SUCCESS) {

            std::cerr
                << "cublasSgetrsBatched failed during benchmark, "
                << "status="
                << static_cast<int>(status)
                << "\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_A_ptr);
            cudaFree(d_A_const_ptr);
            cudaFree(d_B_ptr);
            cudaFree(d_pivot);

            return false;
        }
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));


    /*
     * ============================================================
     * Timing
     * ============================================================
     */

    float total_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &total_ms,
        start,
        stop));

    const double avg_ms =
        static_cast<double>(total_ms)
        / static_cast<double>(kRepeat);


    /*
     * ============================================================
     * FLOPs
     *
     * Approximately:
     *
     *     2 * batch * N^2 * nrhs
     * ============================================================
     */

    const double flops =
        2.0
        * static_cast<double>(batch_size)
        * static_cast<double>(n)
        * static_cast<double>(n)
        * static_cast<double>(nrhs);

    const double gflops =
        flops
        / (avg_ms * 1.0e6);


    std::cout << std::fixed;

    std::cout
        << "Getrs time = "
        << std::setprecision(6)
        << avg_ms
        << " ms\n";

    std::cout
        << "Getrs GFLOPS = "
        << std::setprecision(2)
        << gflops
        << "\n";


    /*
     * ============================================================
     * Copy result
     * ============================================================
     */

    std::vector<float> h_X(total_b);

    CHECK_CUDA(cudaMemcpy(
        h_X.data(),
        d_B,
        total_b * sizeof(float),
        cudaMemcpyDeviceToHost));


    std::cout << "STATUS = PASS\n";


    /*
     * ============================================================
     * Cleanup
     * ============================================================
     */

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_A_ptr));
    CHECK_CUDA(cudaFree(d_A_const_ptr));
    CHECK_CUDA(cudaFree(d_B_ptr));
    CHECK_CUDA(cudaFree(d_pivot));

    return true;
}


int main() {

    int device = 0;

    CHECK_CUDA(cudaSetDevice(device));


    cudaDeviceProp prop{};

    CHECK_CUDA(cudaGetDeviceProperties(
        &prop,
        device));


    std::cout
        << "========================================\n";

    std::cout
        << "GPU: "
        << prop.name
        << "\n";

    std::cout
        << "Compute Capability: "
        << prop.major
        << "."
        << prop.minor
        << "\n";

    std::cout
        << "Global Memory: "
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(prop.totalGlobalMem)
            / (1024.0 * 1024.0 * 1024.0)
        << " GB\n";

    std::cout
        << "========================================\n";


    cublasHandle_t handle;

    CHECK_CUBLAS(
        cublasCreate(&handle));


    /*
     * Large-N tests
     */
    std::vector<TestConfig> tests = {

        {8, 1280, 16},
        {8, 2048, 16},
        {8, 4096, 16},
        {8, 8192, 16},
        {8, 1280, 72},
        {8, 2048, 72},
        {8, 4096, 72},
        {8, 8192, 72},
        {8, 1280, 144},
        {8, 2048, 144},
        {8, 4096, 144},
        {8, 8192, 144},
        {8, 8, 144},
        {8, 8, 1440},
        {8, 8, 14400},
        {8192, 8, 8},
        {8192, 8, 16},
        {8192, 8, 72},
        {8192, 8, 64}

        // // N = 256，nrhs = 1, batch=1
        // {256, 128, 1},
        // {512, 128, 1},
        // {1024, 128, 1},
        // {2048, 128, 1},
        // {4096, 128, 1},
        // {8192, 128, 1},

        // // Multiple RHS
        // {256, 256, 1},
        // {512, 256, 1},
        // {1024, 256, 1},
        // {2048, 256, 1},
        // {4096, 256, 1},
        // {8192, 256, 1},

        // // Large RHS
        // {256, 128, 72},
        // {512, 128, 72},
        // {1024, 128, 72},
        // {2048, 128, 72},
        // {4096, 128, 72},
        // {8192, 128, 72},

        // // Large RHS
        // {256, 128, 144},
        // {512, 128, 144},
        // {1024, 128, 144},
        // {2048, 128, 144},
        // {4096, 128, 144},
        // {8192, 128, 144}
  
    };


    for (const TestConfig& config : tests) {

        RunTest(handle, config);
    }


    CHECK_CUBLAS(
        cublasDestroy(handle));

    CHECK_CUDA(
        cudaDeviceReset());

    return 0;
}