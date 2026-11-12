#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct TestCase {
    std::string name;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int batch_size;
    cublasOperation_t trans;
    bool null_a_array;
    bool null_b_array;
    bool null_ipiv;
    bool null_info;
};

static const char* CublasStatusString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        default:
            return "UNKNOWN";
    }
}

static const char* CudaErrorString(cudaError_t error) {
    return cudaGetErrorString(error);
}

static const char* TransString(cublasOperation_t trans) {
    switch (trans) {
        case CUBLAS_OP_N:
            return "N";
        case CUBLAS_OP_T:
            return "T";
        case CUBLAS_OP_C:
            return "C";
        default:
            return "INVALID";
    }
}

static bool RunTest(const TestCase& test) {
    std::cout << "\n============================================================\n";
    std::cout << "CASE: " << test.name << "\n";
    std::cout << "============================================================\n";

    cublasHandle_t handle = nullptr;

    float* d_a = nullptr;
    float* d_b = nullptr;
    int* d_ipiv = nullptr;

    float** d_a_array = nullptr;
    float** d_b_array = nullptr;

    int* d_info = nullptr;

    auto Cleanup = [&]() {
        if (d_a != nullptr) {
            cudaFree(d_a);
        }

        if (d_b != nullptr) {
            cudaFree(d_b);
        }

        if (d_ipiv != nullptr) {
            cudaFree(d_ipiv);
        }

        if (d_a_array != nullptr) {
            cudaFree(d_a_array);
        }

        if (d_b_array != nullptr) {
            cudaFree(d_b_array);
        }

        if (d_info != nullptr) {
            cudaFree(d_info);
        }

        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    };

    cublasStatus_t status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasCreate: "
                  << CublasStatusString(status)
                  << "\n";
        return false;
    }

    const int n = test.n;
    const int nrhs = test.nrhs;
    const int lda = test.lda;
    const int ldb = test.ldb;
    const int batch_size = test.batch_size;

    /*
     * ------------------------------------------------------------
     * Allocate A
     * ------------------------------------------------------------
     */
    if (n > 0 && lda > 0) {
        const size_t a_elements =
            static_cast<size_t>(lda) *
            static_cast<size_t>(n);

        cudaError_t error = cudaMalloc(
            reinterpret_cast<void**>(&d_a),
            a_elements * sizeof(float));

        if (error != cudaSuccess) {
            std::cout << "cudaMalloc(A): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }

        error = cudaMemset(
            d_a,
            0,
            a_elements * sizeof(float));

        if (error != cudaSuccess) {
            std::cout << "cudaMemset(A): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }
    }

    /*
     * ------------------------------------------------------------
     * Allocate B
     * ------------------------------------------------------------
     */
    if (n > 0 && nrhs > 0 && ldb > 0) {
        const size_t b_elements =
            static_cast<size_t>(ldb) *
            static_cast<size_t>(nrhs);

        cudaError_t error = cudaMalloc(
            reinterpret_cast<void**>(&d_b),
            b_elements * sizeof(float));

        if (error != cudaSuccess) {
            std::cout << "cudaMalloc(B): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }

        error = cudaMemset(
            d_b,
            0,
            b_elements * sizeof(float));

        if (error != cudaSuccess) {
            std::cout << "cudaMemset(B): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }
    }

    /*
     * ------------------------------------------------------------
     * Allocate pivot array
     *
     * devIpiv == NULL is intentionally allowed as a test case.
     * ------------------------------------------------------------
     */
    if (!test.null_ipiv && batch_size > 0 && n > 0) {
        const size_t ipiv_elements =
            static_cast<size_t>(n) *
            static_cast<size_t>(batch_size);

        cudaError_t error = cudaMalloc(
            reinterpret_cast<void**>(&d_ipiv),
            ipiv_elements * sizeof(int));

        if (error != cudaSuccess) {
            std::cout << "cudaMalloc(ipiv): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }

        error = cudaMemset(
            d_ipiv,
            0,
            ipiv_elements * sizeof(int));

        if (error != cudaSuccess) {
            std::cout << "cudaMemset(ipiv): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }
    }

    /*
     * ------------------------------------------------------------
     * Allocate A pointer array
     * ------------------------------------------------------------
     */
    if (!test.null_a_array && batch_size > 0) {
        cudaError_t error = cudaMalloc(
            reinterpret_cast<void**>(&d_a_array),
            static_cast<size_t>(batch_size) *
                sizeof(float*));

        if (error != cudaSuccess) {
            std::cout << "cudaMalloc(Aarray): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }

        std::vector<float*> host_a_array(
            static_cast<size_t>(batch_size),
            d_a);

        error = cudaMemcpy(
            d_a_array,
            host_a_array.data(),
            static_cast<size_t>(batch_size) *
                sizeof(float*),
            cudaMemcpyHostToDevice);

        if (error != cudaSuccess) {
            std::cout << "cudaMemcpy(Aarray): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }
    }

    /*
     * ------------------------------------------------------------
     * Allocate B pointer array
     * ------------------------------------------------------------
     */
    if (!test.null_b_array && batch_size > 0) {
        cudaError_t error = cudaMalloc(
            reinterpret_cast<void**>(&d_b_array),
            static_cast<size_t>(batch_size) *
                sizeof(float*));

        if (error != cudaSuccess) {
            std::cout << "cudaMalloc(Barray): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }

        std::vector<float*> host_b_array(
            static_cast<size_t>(batch_size),
            d_b);

        error = cudaMemcpy(
            d_b_array,
            host_b_array.data(),
            static_cast<size_t>(batch_size) *
                sizeof(float*),
            cudaMemcpyHostToDevice);

        if (error != cudaSuccess) {
            std::cout << "cudaMemcpy(Barray): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }
    }

    /*
     * ------------------------------------------------------------
     * Allocate info
     *
     * info == NULL is intentionally allowed as a test case.
     * ------------------------------------------------------------
     */
    if (!test.null_info) {
        cudaError_t error = cudaMalloc(
            reinterpret_cast<void**>(&d_info),
            sizeof(int));

        if (error != cudaSuccess) {
            std::cout << "cudaMalloc(info): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }

        error = cudaMemset(
            d_info,
            0,
            sizeof(int));

        if (error != cudaSuccess) {
            std::cout << "cudaMemset(info): "
                      << CudaErrorString(error)
                      << "\n";

            Cleanup();
            return false;
        }
    }

    /*
     * 清除此前可能存在的 CUDA error。
     */
    cudaGetLastError();

    /*
     * ============================================================
     * cuBLAS GOLDEN CALL
     * ============================================================
     */
    status = cublasSgetrsBatched(
        handle,
        test.trans,
        n,
        nrhs,
        reinterpret_cast<const float* const*>(
            d_a_array),
        lda,
        d_ipiv,
        d_b_array,
        ldb,
        d_info,
        batch_size);

    /*
     * ------------------------------------------------------------
     * Read info
     * ------------------------------------------------------------
     */
    int host_info = 0;
    cudaError_t info_copy_error = cudaSuccess;

    if (d_info != nullptr) {
        info_copy_error = cudaMemcpy(
            &host_info,
            d_info,
            sizeof(int),
            cudaMemcpyDeviceToHost);
    }

    /*
     * ------------------------------------------------------------
     * Synchronize
     *
     * getrsBatched is asynchronous. Therefore API SUCCESS does
     * not necessarily mean the kernel executed successfully.
     * ------------------------------------------------------------
     */
    cudaError_t sync_error = cudaDeviceSynchronize();

    /*
     * Synchronization succeeds时再重新读取 info。
     */
    if (d_info != nullptr &&
        sync_error == cudaSuccess) {
        info_copy_error = cudaMemcpy(
            &host_info,
            d_info,
            sizeof(int),
            cudaMemcpyDeviceToHost);
    }

    /*
     * ------------------------------------------------------------
     * Print result
     * ------------------------------------------------------------
     */
    std::cout << std::left
              << std::setw(24) << "Parameter"
              << std::setw(20) << "Value"
              << "\n";

    std::cout << std::setw(24)
              << "n"
              << std::setw(20)
              << n
              << "\n";

    std::cout << std::setw(24)
              << "nrhs"
              << std::setw(20)
              << nrhs
              << "\n";

    std::cout << std::setw(24)
              << "lda"
              << std::setw(20)
              << lda
              << "\n";

    std::cout << std::setw(24)
              << "ldb"
              << std::setw(20)
              << ldb
              << "\n";

    std::cout << std::setw(24)
              << "batch_size"
              << std::setw(20)
              << batch_size
              << "\n";

    std::cout << std::setw(24)
              << "trans"
              << std::setw(20)
              << TransString(test.trans)
              << "\n";

    std::cout << "\n";

    std::cout << std::setw(24)
              << "cuBLAS status"
              << std::setw(20)
              << CublasStatusString(status)
              << "\n";

    std::cout << std::setw(24)
              << "info"
              << std::setw(20);

    if (test.null_info) {
        std::cout << "NULL";
    } else if (info_copy_error != cudaSuccess) {
        std::cout << "COPY_ERROR";
    } else {
        std::cout << host_info;
    }

    std::cout << "\n";

    std::cout << std::setw(24)
              << "CUDA synchronize"
              << std::setw(20)
              << CudaErrorString(sync_error)
              << "\n";

    /*
     * ------------------------------------------------------------
     * Observation
     * ------------------------------------------------------------
     */
    if (status == CUBLAS_STATUS_SUCCESS &&
        sync_error != cudaSuccess) {
        std::cout
            << "\nOBSERVE: cuBLAS API returned SUCCESS, "
               "but asynchronous CUDA execution failed.\n";
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout
            << "\nOBSERVE: cuBLAS returned "
            << CublasStatusString(status)
            << ".\n";
    }

    if (sync_error != cudaSuccess) {
        std::cout
            << "OBSERVE: CUDA execution error = "
            << CudaErrorString(sync_error)
            << ".\n";
    }

    if (test.null_info) {
        std::cout
            << "OBSERVE: info = NULL.\n";
    }

    Cleanup();

    return true;
}

static std::vector<TestCase> BuildTestCases() {
    return {
        {
            "normal",
            4, 2, 4, 4, 2,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "n_zero",
            0, 1, 1, 1, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "n_one",
            1, 1, 1, 1, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "n_negative",
            -1, 1, 1, 1, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "nrhs_zero",
            4, 0, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "nrhs_one",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "nrhs_negative",
            4, -1, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "lda_zero",
            4, 1, 0, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "lda_n_minus_1",
            4, 1, 3, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "lda_min",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "lda_n_plus_1",
            4, 1, 5, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "ldb_zero",
            4, 1, 4, 0, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "ldb_n_minus_1",
            4, 1, 4, 3, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "ldb_min",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "ldb_n_plus_1",
            4, 1, 4, 5, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "trans_N",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "trans_T",
            4, 1, 4, 4, 1,
            CUBLAS_OP_T,
            false, false, false, false
        },

        {
            "trans_C",
            4, 1, 4, 4, 1,
            CUBLAS_OP_C,
            false, false, false, false
        },

        {
            "trans_invalid",
            4, 1, 4, 4, 1,
            static_cast<cublasOperation_t>(999),
            false, false, false, false
        },

        {
            "batch_zero_null_arrays",
            4, 1, 4, 4, 0,
            CUBLAS_OP_N,
            true, true, false, false
        },

        {
            "batch_zero_nonnull_arrays",
            4, 1, 4, 4, 0,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "batch_one",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "batch_two",
            4, 1, 4, 4, 2,
            CUBLAS_OP_N,
            false, false, false, false
        },

        {
            "batch_negative",
            4, 1, 4, 4, -1,
            CUBLAS_OP_N,
            true, true, false, false
        },

        {
            "ipiv_null",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, true, false
        },

        {
            "Aarray_null",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            true, false, false, false
        },

        {
            "Barray_null",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            false, true, false, false
        },

        {
            "info_null",
            4, 1, 4, 4, 1,
            CUBLAS_OP_N,
            false, false, false, true
        },
    };
}

int main(int argc, char** argv) {
    const int device_id = 0;

    cudaDeviceProp prop{};

    cudaError_t error =
        cudaGetDeviceProperties(&prop, device_id);

    if (error != cudaSuccess) {
        std::cerr
            << "cudaGetDeviceProperties failed: "
            << cudaGetErrorString(error)
            << "\n";

        return EXIT_FAILURE;
    }

    cudaSetDevice(device_id);

    std::cout
        << "============================================================\n"
        << " cuBLAS getrsBatched Golden Test\n"
        << "============================================================\n";

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

    const std::vector<TestCase> tests =
        BuildTestCases();

    /*
     * 支持单独执行某个 case：
     *
     * ./getrs_bendtest_cu Aarray_null
     */
    if (argc > 1) {
        const std::string requested = argv[1];

        for (const TestCase& test : tests) {
            if (test.name == requested) {
                RunTest(test);
                return EXIT_SUCCESS;
            }
        }

        std::cerr
            << "Unknown test case: "
            << requested
            << "\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "Number of test cases: "
        << tests.size()
        << "\n";

    for (const TestCase& test : tests) {
        RunTest(test);
    }

    std::cout
        << "\n============================================================\n"
        << " Golden Test Finished\n"
        << "============================================================\n";

    return EXIT_SUCCESS;
}