#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// nvcc getrs_bendtest_cu.cu -o getrs_bendtest_cu -lcublas

// ============================================================
// Configuration
// ============================================================

using DataType = float;

constexpr int kNormalN = 4;
constexpr int kNormalNrhs = 2;
constexpr int kNormalBatchSize = 2;

// ============================================================
// Error helpers
// ============================================================

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

// ============================================================
// Test case
// ============================================================

enum class Expectation {
    CompareWithCuBLAS,
    MustBeInvalidValue,
    ObserveOnly
};

struct TestCase {
    std::string name;

    cublasOperation_t trans = CUBLAS_OP_N;

    int n = 4;
    int nrhs = 1;
    int lda = 4;
    int ldb = 4;
    int batchSize = 1;

    bool nullAarray = false;
    bool nullBarray = false;
    bool nullIpiv = false;
    bool nullInfo = false;

    // If true, A/B pointer arrays are allocated even when batchSize == 0.
    bool allocatePointerArraysForZeroBatch = false;

    Expectation expectation = Expectation::CompareWithCuBLAS;
};

// ============================================================
// Device memory holder
// ============================================================

template <typename T>
struct DeviceBatch {
    std::vector<T*> d_A;
    std::vector<T*> d_B;

    T** d_Aarray = nullptr;
    T** d_Barray = nullptr;

    int* d_Ipiv = nullptr;

    std::vector<T*> allocatedA;
    std::vector<T*> allocatedB;

    int n = 0;
    int nrhs = 0;
    int lda = 0;
    int ldb = 0;
    int batchSize = 0;

    ~DeviceBatch() {
        Release();
    }

    void Release() {
        for (T* p : allocatedA) {
            if (p != nullptr) {
                cudaFree(p);
            }
        }

        for (T* p : allocatedB) {
            if (p != nullptr) {
                cudaFree(p);
            }
        }

        if (d_Aarray != nullptr) {
            cudaFree(d_Aarray);
        }

        if (d_Barray != nullptr) {
            cudaFree(d_Barray);
        }

        if (d_Ipiv != nullptr) {
            cudaFree(d_Ipiv);
        }

        allocatedA.clear();
        allocatedB.clear();
        d_A.clear();
        d_B.clear();

        d_Aarray = nullptr;
        d_Barray = nullptr;
        d_Ipiv = nullptr;
    }
};

// ============================================================
// Allocate a valid normal batch
// ============================================================

static bool CreateValidBatch(
    int n,
    int nrhs,
    int lda,
    int ldb,
    int batchSize,
    bool allocatePointerArraysForZeroBatch,
    DeviceBatch<DataType>& batch) {

    batch.n = n;
    batch.nrhs = nrhs;
    batch.lda = lda;
    batch.ldb = ldb;
    batch.batchSize = batchSize;

    if (batchSize > 0) {
        batch.d_A.resize(batchSize, nullptr);
        batch.d_B.resize(batchSize, nullptr);

        for (int i = 0; i < batchSize; ++i) {
            size_t aElements =
                static_cast<size_t>(std::max(1, lda)) *
                static_cast<size_t>(std::max(1, n));

            size_t bElements =
                static_cast<size_t>(std::max(1, ldb)) *
                static_cast<size_t>(std::max(1, nrhs));

            DataType* dA = nullptr;
            DataType* dB = nullptr;

            cudaError_t err = cudaMalloc(
                reinterpret_cast<void**>(&dA),
                aElements * sizeof(DataType));

            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc(A) failed: "
                          << CudaErrorString(err) << "\n";
                return false;
            }

            err = cudaMalloc(
                reinterpret_cast<void**>(&dB),
                bElements * sizeof(DataType));

            if (err != cudaSuccess) {
                cudaFree(dA);

                std::cerr << "cudaMalloc(B) failed: "
                          << CudaErrorString(err) << "\n";
                return false;
            }

            batch.allocatedA.push_back(dA);
            batch.allocatedB.push_back(dB);

            batch.d_A[i] = dA;
            batch.d_B[i] = dB;

            // Initialize memory.
            cudaMemset(
                dA,
                0,
                aElements * sizeof(DataType));

            cudaMemset(
                dB,
                0,
                bElements * sizeof(DataType));
        }

        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&batch.d_Aarray),
            batchSize * sizeof(DataType*));

        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc(Aarray) failed: "
                      << CudaErrorString(err) << "\n";
            return false;
        }

        err = cudaMalloc(
            reinterpret_cast<void**>(&batch.d_Barray),
            batchSize * sizeof(DataType*));

        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc(Barray) failed: "
                      << CudaErrorString(err) << "\n";
            return false;
        }

        cudaMemcpy(
            batch.d_Aarray,
            batch.d_A.data(),
            batchSize * sizeof(DataType*),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            batch.d_Barray,
            batch.d_B.data(),
            batchSize * sizeof(DataType*),
            cudaMemcpyHostToDevice);

        // Pivot array: n * batchSize
        if (n > 0) {
            err = cudaMalloc(
                reinterpret_cast<void**>(&batch.d_Ipiv),
                static_cast<size_t>(n) *
                    static_cast<size_t>(batchSize) *
                    sizeof(int));

            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc(Ipiv) failed: "
                          << CudaErrorString(err) << "\n";
                return false;
            }

            // We don't need valid pivots for parameter-error cases.
            // For normal numerical test, they will be overwritten by getrf.
            cudaMemset(
                batch.d_Ipiv,
                0,
                static_cast<size_t>(n) *
                    static_cast<size_t>(batchSize) *
                    sizeof(int));
        }
    } else if (allocatePointerArraysForZeroBatch) {
        // Explicitly allocate non-null pointer arrays for batchSize == 0.
        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&batch.d_Aarray),
            sizeof(DataType*));

        if (err != cudaSuccess) {
            return false;
        }

        err = cudaMalloc(
            reinterpret_cast<void**>(&batch.d_Barray),
            sizeof(DataType*));

        if (err != cudaSuccess) {
            return false;
        }
    }

    return true;
}

// ============================================================
// Generate a mathematically valid LU + B
//
// A = [ 4 1 2 0
//       0 3 1 1
//       1 0 5 2
//       0 2 0 6 ]
//
// B is generated from a known X:
// X = [1 2
//      2 1
//      1 3
//      2 4]
//
// We first calculate B = A * X, then factor A with getrf.
// ============================================================

static bool PrepareValidSystem(
    cublasHandle_t handle,
    DeviceBatch<DataType>& batch) {

    if (batch.batchSize <= 0 || batch.n != 4 || batch.nrhs != 2) {
        return true;
    }

    const int n = 4;
    const int nrhs = 2;

    std::vector<DataType> hA = {
        4, 0, 1, 0,
        1, 3, 0, 2,
        2, 1, 5, 0,
        0, 1, 2, 6
    };

    // Column-major X:
    //
    // X =
    // [1 2
    //  2 1
    //  1 3
    //  2 4]
    //
    std::vector<DataType> hX = {
        1, 2, 1, 2,
        2, 1, 3, 4
    };

    // B = A * X on host, column-major.
    std::vector<DataType> hB(n * nrhs, 0);

    for (int col = 0; col < nrhs; ++col) {
        for (int row = 0; row < n; ++row) {
            DataType value = 0;

            for (int k = 0; k < n; ++k) {
                value += hA[row + k * n] *
                         hX[k + col * n];
            }

            hB[row + col * n] = value;
        }
    }

    for (int i = 0; i < batch.batchSize; ++i) {
        cudaMemcpy(
            batch.d_A[i],
            hA.data(),
            hA.size() * sizeof(DataType),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            batch.d_B[i],
            hB.data(),
            hB.size() * sizeof(DataType),
            cudaMemcpyHostToDevice);
    }

    // LU factorization.
    std::vector<int> hInfo(batch.batchSize, -999);

    int** dummy = nullptr;
    (void)dummy;

    // Need device pointer to pointer array and device info.
    int* dInfo = nullptr;

    cudaError_t cudaStatus = cudaMalloc(
        reinterpret_cast<void**>(&dInfo),
        batch.batchSize * sizeof(int));

    if (cudaStatus != cudaSuccess) {
        return false;
    }

    cublasStatus_t status =
        cublasSgetrfBatched(
            handle,
            n,
            batch.d_Aarray,
            batch.lda,
            batch.d_Ipiv,
            dInfo,
            batch.batchSize);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgetrfBatched failed: "
                  << CublasStatusString(status) << "\n";
        cudaFree(dInfo);
        return false;
    }

    cudaMemcpy(
        hInfo.data(),
        dInfo,
        batch.batchSize * sizeof(int),
        cudaMemcpyDeviceToHost);

    cudaFree(dInfo);

    for (int i = 0; i < batch.batchSize; ++i) {
        if (hInfo[i] != 0) {
            std::cerr << "getrf batch[" << i
                      << "] info = " << hInfo[i] << "\n";
            return false;
        }
    }

    return true;
}

// ============================================================
// Result
// ============================================================

struct CallResult {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    int info = 0;
    bool infoValid = false;

    cudaError_t cudaError = cudaSuccess;

    bool callCompleted = false;

    std::string statusString;
    std::string cudaErrorString;
};

struct ComparisonResult {
    bool pass = false;
    std::string reason;
};

// ============================================================
// cuBLAS call
// ============================================================

static CallResult RunCuBLAS(
    cublasHandle_t handle,
    const TestCase& tc) {

    CallResult result;

    DeviceBatch<DataType> batch;

    bool allocateArrays =
        tc.allocatePointerArraysForZeroBatch;

    if (tc.batchSize > 0) {
        allocateArrays = true;
    }

    if (!CreateValidBatch(
            tc.n,
            tc.nrhs,
            tc.lda,
            tc.ldb,
            tc.batchSize,
            allocateArrays,
            batch)) {

        result.cudaError = cudaGetLastError();
        result.cudaErrorString =
            CudaErrorString(result.cudaError);

        return result;
    }

    // Prepare valid LU only for the normal numerical case.
    if (tc.name == "normal") {
        if (!PrepareValidSystem(handle, batch)) {
            result.status = CUBLAS_STATUS_INTERNAL_ERROR;
            result.statusString =
                "Failed to prepare numerical system";
            return result;
        }
    }

    const DataType* const* Aarray = nullptr;
    DataType* const* Barray = nullptr;

    if (!tc.nullAarray) {
        Aarray = batch.d_Aarray;
    }

    if (!tc.nullBarray) {
        Barray = batch.d_Barray;
    }

    const int* dIpiv =
        tc.nullIpiv ? nullptr : batch.d_Ipiv;

    int info = 0x12345678;

    int* infoPtr =
        tc.nullInfo ? nullptr : &info;

    // --------------------------------------------------------
    // Clear previous CUDA error.
    // --------------------------------------------------------

    cudaGetLastError();

    result.status =
        cublasSgetrsBatched(
            handle,
            tc.trans,
            tc.n,
            tc.nrhs,
            Aarray,
            tc.lda,
            dIpiv,
            Barray,
            tc.ldb,
            infoPtr,
            tc.batchSize);

    result.statusString =
        CublasStatusString(result.status);

    result.info = info;
    result.infoValid = !tc.nullInfo;

    // Synchronize to expose asynchronous execution failures.
    cudaError_t syncError = cudaDeviceSynchronize();

    result.cudaError = syncError;
    result.cudaErrorString =
        CudaErrorString(syncError);

    result.callCompleted = true;

    return result;
}



// ============================================================
// Comparison
// ============================================================

static ComparisonResult CompareResults(
    const TestCase& tc,
    const CallResult& cudaResult,
    const CallResult& zxResult) {

    ComparisonResult result;

    // Observation-only cases:
    if (tc.expectation == Expectation::ObserveOnly) {
        result.pass = true;
        result.reason = "OBSERVE_ONLY";
        return result;
    }

    // Compare status.
    if (cudaResult.status != zxResult.status) {
        std::ostringstream oss;

        oss << "status mismatch: cuBLAS="
            << CublasStatusString(zxResult.status);

        result.reason = oss.str();
        return result;
    }

    // If both return SUCCESS, compare info.
    if (cudaResult.status == CUBLAS_STATUS_SUCCESS &&
        cudaResult.infoValid &&
        zxResult.infoValid) {

        if (cudaResult.info != zxResult.info) {
            std::ostringstream oss;

            oss << "info mismatch: cuBLAS="
                << cudaResult.info
                << zxResult.info;

            result.reason = oss.str();
            return result;
        }
    }

    result.pass = true;
    result.reason = "MATCH";

    return result;
}

// ============================================================
// Test case generation
// ============================================================

static std::vector<TestCase> GenerateTests() {

    std::vector<TestCase> tests;

    // --------------------------------------------------------
    // 1. Normal
    // --------------------------------------------------------

    tests.push_back({
        "normal",
        CUBLAS_OP_N,
        4,
        2,
        4,
        4,
        2,
        false,
        false,
        false,
        false,
        false,
        Expectation::CompareWithCuBLAS
    });

    // --------------------------------------------------------
    // 2. n boundary
    // --------------------------------------------------------

    tests.push_back({
        "n_zero",
        CUBLAS_OP_N,
        0,
        1,
        1,
        1,
        1
    });

    tests.push_back({
        "n_one",
        CUBLAS_OP_N,
        1,
        1,
        1,
        1,
        1
    });

    tests.push_back({
        "n_negative",
        CUBLAS_OP_N,
        -1,
        1,
        1,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
        Expectation::CompareWithCuBLAS
    });

    // --------------------------------------------------------
    // 3. nrhs boundary
    // --------------------------------------------------------

    tests.push_back({
        "nrhs_zero",
        CUBLAS_OP_N,
        4,
        0,
        4,
        4,
        1
    });

    tests.push_back({
        "nrhs_one",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1
    });

    tests.push_back({
        "nrhs_negative",
        CUBLAS_OP_N,
        4,
        -1,
        4,
        4,
        1
    });

    // --------------------------------------------------------
    // 4. lda boundary
    // --------------------------------------------------------

    tests.push_back({
        "lda_zero",
        CUBLAS_OP_N,
        4,
        1,
        0,
        4,
        1
    });

    tests.push_back({
        "lda_n_minus_1",
        CUBLAS_OP_N,
        4,
        1,
        3,
        4,
        1
    });

    tests.push_back({
        "lda_min",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1
    });

    tests.push_back({
        "lda_n_plus_1",
        CUBLAS_OP_N,
        4,
        1,
        5,
        4,
        1
    });

    // --------------------------------------------------------
    // 5. ldb boundary
    // --------------------------------------------------------

    tests.push_back({
        "ldb_zero",
        CUBLAS_OP_N,
        4,
        1,
        4,
        0,
        1
    });

    tests.push_back({
        "ldb_n_minus_1",
        CUBLAS_OP_N,
        4,
        1,
        4,
        3,
        1
    });

    tests.push_back({
        "ldb_min",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1
    });

    tests.push_back({
        "ldb_n_plus_1",
        CUBLAS_OP_N,
        4,
        1,
        4,
        5,
        1
    });

    // --------------------------------------------------------
    // 6. trans
    // --------------------------------------------------------

    tests.push_back({
        "trans_N",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1
    });

    tests.push_back({
        "trans_T",
        CUBLAS_OP_T,
        4,
        1,
        4,
        4,
        1
    });

    tests.push_back({
        "trans_C",
        CUBLAS_OP_C,
        4,
        1,
        4,
        4,
        1
    });

    tests.push_back({
        "trans_invalid",
        static_cast<cublasOperation_t>(999),
        4,
        1,
        4,
        4,
        1
    });

    // --------------------------------------------------------
    // 7. batchSize
    // --------------------------------------------------------

    tests.push_back({
        "batch_zero_null_arrays",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        0,
        true,
        true,
        true,
        false,
        false,
        Expectation::CompareWithCuBLAS
    });

    tests.push_back({
        "batch_zero_nonnull_arrays",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        0,
        false,
        false,
        true,
        false,
        true,
        Expectation::CompareWithCuBLAS
    });

    tests.push_back({
        "batch_one",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1
    });

    tests.push_back({
        "batch_two",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        2
    });

    tests.push_back({
        "batch_negative",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        -1
    });

    // --------------------------------------------------------
    // 8. Pivot
    // --------------------------------------------------------

    tests.push_back({
        "ipiv_null",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1,
        false,
        false,
        true,
        false,
        false,
        Expectation::CompareWithCuBLAS
    });

    // --------------------------------------------------------
    // 9. NULL A/B array
    //
    // These are observation cases because NVIDIA does not
    // document NULL Aarray/Barray as a normal legal parameter.
    // --------------------------------------------------------

    tests.push_back({
        "Aarray_null_batch1",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1,
        true,
        false,
        false,
        false,
        false,
        Expectation::ObserveOnly
    });

    tests.push_back({
        "Barray_null_batch1",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1,
        false,
        true,
        false,
        false,
        false,
        Expectation::ObserveOnly
    });

    // --------------------------------------------------------
    // 10. info == NULL
    // --------------------------------------------------------

    tests.push_back({
        "info_null",
        CUBLAS_OP_N,
        4,
        1,
        4,
        4,
        1,
        false,
        false,
        false,
        true,
        false,
        Expectation::ObserveOnly
    });

    return tests;
}

// ============================================================
// Pretty print
// ============================================================

static std::string ExpectationString(Expectation e) {
    switch (e) {
        case Expectation::CompareWithCuBLAS:
            return "COMPARE";
        case Expectation::MustBeInvalidValue:
            return "INVALID";
        case Expectation::ObserveOnly:
            return "OBSERVE";
        default:
            return "UNKNOWN";
    }
}

// ============================================================
// Main
// ============================================================

int main() {

    std::cout
        << "============================================================\n"
        << " cuBLAS / zxBLAS getrsBatched Boundary Test\n"
        << "============================================================\n";

    int deviceCount = 0;

    cudaError_t cudaStatus =
        cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        std::cerr
            << "No CUDA device: "
            << CudaErrorString(cudaStatus)
            << "\n";
        return 1;
    }

    cudaDeviceProp prop{};

    cudaGetDeviceProperties(&prop, 0);

    std::cout
        << "GPU: " << prop.name << "\n"
        << "Compute Capability: "
        << prop.major << "." << prop.minor << "\n\n";

    // --------------------------------------------------------
    // Create cuBLAS handle
    // --------------------------------------------------------

    cublasHandle_t handle = nullptr;

    cublasStatus_t status =
        cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr
            << "cublasCreate failed: "
            << CublasStatusString(status)
            << "\n";
        return 1;
    }

    // --------------------------------------------------------
    // Generate tests
    // --------------------------------------------------------

    std::vector<TestCase> tests =
        GenerateTests();

    std::cout
        << "Number of test cases: "
        << tests.size()
        << "\n\n";

    // --------------------------------------------------------
    // Table header
    // --------------------------------------------------------

    std::cout
        << std::left
        << std::setw(30) << "CASE"
        << std::setw(8)  << "n"
        << std::setw(8)  << "nrhs"
        << std::setw(8)  << "lda"
        << std::setw(8)  << "ldb"
        << std::setw(10) << "batch"
        << std::setw(20) << "cuBLAS"
        << std::setw(12) << "info"
        << std::setw(20) << "zxBLAS"
        << std::setw(10) << "RESULT"
        << "\n";

    std::cout
        << std::string(134, '-')
        << "\n";

    int passCount = 0;
    int failCount = 0;
    int observeCount = 0;

    // --------------------------------------------------------
    // Run tests
    // --------------------------------------------------------

    for (const TestCase& tc : tests) {

        CallResult cuResult =
            RunCuBLAS(handle, tc);

        CallResult zxResult =
            RunZXBLAS(handle, tc);

        ComparisonResult comparison =
            CompareResults(
                tc,
                cuResult,
                zxResult);

        if (tc.expectation == Expectation::ObserveOnly) {
            ++observeCount;
        } else if (comparison.pass) {
            ++passCount;
        } else {
            ++failCount;
        }

        std::ostringstream cuInfo;

        if (cuResult.infoValid) {
            cuInfo << cuResult.info;
        } else {
            cuInfo << "NULL";
        }

        std::cout
            << std::left
            << std::setw(30) << tc.name
            << std::setw(8)  << tc.n
            << std::setw(8)  << tc.nrhs
            << std::setw(8)  << tc.lda
            << std::setw(8)  << tc.ldb
            << std::setw(10) << tc.batchSize
            << std::setw(20) << cuResult.statusString
            << std::setw(12) << cuInfo.str()
            << std::setw(20) << zxResult.statusString
            << std::setw(10)
            << (comparison.pass ? "PASS" : "FAIL")
            << "\n";

        if (!comparison.pass) {
            std::cout
                << "    reason: "
                << comparison.reason
                << "\n";
        }

        // Print asynchronous CUDA errors.
        if (cuResult.cudaError != cudaSuccess) {
            std::cout
                << "    cuBLAS CUDA error: "
                << cuResult.cudaErrorString
                << "\n";
        }

        if (zxResult.cudaError != cudaSuccess) {
            std::cout
                << "    zxBLAS CUDA error: "
                << zxResult.cudaErrorString
                << "\n";
        }
    }

    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------

    std::cout
        << "\n============================================================\n"
        << " Summary\n"
        << "============================================================\n";

    std::cout
        << "PASS     : " << passCount << "\n"
        << "FAIL     : " << failCount << "\n"
        << "OBSERVE  : " << observeCount << "\n"
        << "TOTAL    : " << tests.size() << "\n";

    cublasDestroy(handle);

    return failCount == 0 ? 0 : 2;
}