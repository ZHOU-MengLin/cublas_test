#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>

const char* cublas_status_str(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
        default:                             return "UNKNOWN";
    }
}

// ---------- 类型派发 ----------
template<typename T> struct cublas_type;

template<> struct cublas_type<float> {
    static cublasStatus_t getrf(cublasHandle_t h, int n, float** A, int lda,
                                int* piv, int* info, int batch) {
        return cublasSgetrfBatched(h, n, A, lda, piv, info, batch);
    }
    static cublasStatus_t getrs(cublasHandle_t h, cublasOperation_t op,
                                int n, int nrhs, const float** A, int lda,
                                const int* piv, float** B, int ldb,
                                int* info, int batch) {
        return cublasSgetrsBatched(h, op, n, nrhs, A, lda, piv, B, ldb,
                                   info, batch);
    }
};

template<> struct cublas_type<double> {
    static cublasStatus_t getrf(cublasHandle_t h, int n, double** A, int lda,
                                int* piv, int* info, int batch) {
        return cublasDgetrfBatched(h, n, A, lda, piv, info, batch);
    }
    static cublasStatus_t getrs(cublasHandle_t h, cublasOperation_t op,
                                int n, int nrhs, const double** A, int lda,
                                const int* piv, double** B, int ldb,
                                int* info, int batch) {
        return cublasDgetrsBatched(h, op, n, nrhs, A, lda, piv, B, ldb,
                                   info, batch);
    }
};
// --------------------------------

template<typename T>
__global__ void set_identity_kernel(T* A, int n, int lda) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        A[(size_t)j * lda + j] = T(1);
    }
}

template<typename T>
__global__ void set_ones_kernel(T* B, size_t total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) B[idx] = T(1);
}

template<typename T>
bool test_getrs_batched(int n, int nrhs, int batchSize, bool verify) {
    const char* type_name = (sizeof(T) == 4) ? "float" : "double";
    std::cout << "--- n=" << n << " nrhs=" << nrhs
              << " batch=" << batchSize
              << " type=" << type_name << " ---" << std::endl;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    int lda = n;
    int ldb = n;
    size_t a_size = (size_t)lda * (size_t)n * sizeof(T);
    size_t b_size = (size_t)ldb * (size_t)nrhs * sizeof(T);
    size_t ptr_size = (size_t)batchSize * sizeof(T*);
    size_t piv_size = (size_t)n * (size_t)batchSize * sizeof(int);
    size_t info_size = (size_t)batchSize * sizeof(int);

    size_t needed = (size_t)batchSize * (a_size + b_size)
                  + 2 * ptr_size + piv_size + info_size;

    std::cout << "    needed=" << needed / (1024.0 * 1024.0) << " MB"
              << " free=" << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;

    if ((double)needed > (double)free_mem * 0.9) {
        std::cout << "    SKIP: not enough free device memory" << std::endl;
        return false;
    }

    cublasHandle_t handle = nullptr;
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cout << "    cublasCreate failed: " << cublas_status_str(st) << std::endl;
        return false;
    }

    std::vector<T*> hA(batchSize, nullptr), hB(batchSize, nullptr);
    T** dA = nullptr;
    T** dB = nullptr;
    int* dIpiv = nullptr;
    int* dInfo = nullptr;

    bool ok = false;
    bool alloc_failed = false;

    for (int i = 0; i < batchSize; ++i) {
        cudaError_t e1 = cudaMalloc(&hA[i], a_size);
        cudaError_t e2 = cudaMalloc(&hB[i], b_size);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            std::cout << "    cudaMalloc failed at batch=" << i
                      << ": " << cudaGetErrorString(e1) << " / "
                      << cudaGetErrorString(e2) << std::endl;
            alloc_failed = true;
            break;
        }
        cudaMemset(hA[i], 0, a_size);
        cudaMemset(hB[i], 0, b_size);

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        set_identity_kernel<T><<<blocks, threads>>>(hA[i], n, lda);

        size_t total_elems = (size_t)n * (size_t)nrhs;
        int blocks_b = (int)((total_elems + (size_t)threads - 1) / (size_t)threads);
        set_ones_kernel<T><<<blocks_b, threads>>>(hB[i], total_elems);
    }

    if (!alloc_failed) {
        cudaError_t ce = cudaDeviceSynchronize();
        if (ce != cudaSuccess) {
            std::cout << "    init kernel failed: " << cudaGetErrorString(ce) << std::endl;
            alloc_failed = true;
        }
    }

    if (!alloc_failed) {
        cudaError_t e1 = cudaMalloc(&dA, ptr_size);
        cudaError_t e2 = cudaMalloc(&dB, ptr_size);
        cudaError_t e3 = cudaMalloc(&dIpiv, piv_size);
        cudaError_t e4 = cudaMalloc(&dInfo, info_size);
        if (e1 != cudaSuccess || e2 != cudaSuccess ||
            e3 != cudaSuccess || e4 != cudaSuccess) {
            std::cout << "    cudaMalloc (arrays) failed: "
                      << cudaGetErrorString(e1) << " / "
                      << cudaGetErrorString(e2) << " / "
                      << cudaGetErrorString(e3) << " / "
                      << cudaGetErrorString(e4) << std::endl;
            alloc_failed = true;
        }
    }

    if (!alloc_failed) {
        cudaMemcpy(dA, hA.data(), ptr_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), ptr_size, cudaMemcpyHostToDevice);

        // 1) getrfBatched
        st = cublas_type<T>::getrf(handle, n, dA, lda, dIpiv, dInfo, batchSize);
        cudaError_t ce = cudaDeviceSynchronize();
        std::cout << "    getrf: cublas=" << cublas_status_str(st)
                  << " cuda=" << cudaGetErrorString(ce) << std::endl;

        bool getrf_ok = (st == CUBLAS_STATUS_SUCCESS && ce == cudaSuccess);

        std::vector<int> hInfo(batchSize, 0);
        if (getrf_ok &&
            cudaMemcpy(hInfo.data(), dInfo, info_size, cudaMemcpyDeviceToHost)
                == cudaSuccess) {
            int bad = 0;
            for (int i = 0; i < batchSize; ++i) {
                if (hInfo[i] != 0) {
                    if (bad < 5)
                        std::cout << "    getrf info[" << i << "]=" << hInfo[i]
                                  << std::endl;
                    ++bad;
                }
            }
            if (bad > 0) {
                std::cout << "    getrf: " << bad
                          << " batches had singular pivot" << std::endl;
                getrf_ok = false;
            }
        }

        if (getrf_ok) {
            // 2) getrsBatched
            st = cublas_type<T>::getrs(handle, CUBLAS_OP_N, n, nrhs,
                                       (const T**)dA, lda, dIpiv,
                                       (const T**)dB, ldb, dInfo, batchSize);
            ce = cudaDeviceSynchronize();
            std::cout << "    getrs: cublas=" << cublas_status_str(st)
                      << " cuda=" << cudaGetErrorString(ce) << std::endl;

            ok = (st == CUBLAS_STATUS_SUCCESS && ce == cudaSuccess);

            if (ok && verify) {
                std::vector<T> hB0((size_t)n * (size_t)nrhs);
                cudaError_t ce2 = cudaMemcpy(hB0.data(), hB[0], b_size,
                                             cudaMemcpyDeviceToHost);
                if (ce2 == cudaSuccess) {
                    bool verify_ok = true;
                    for (size_t k = 0; k < hB0.size(); ++k) {
                        if (hB0[k] != T(1)) { verify_ok = false; break; }
                    }
                    std::cout << "    verify: "
                              << (verify_ok ? "OK" : "FAILED") << std::endl;
                } else {
                    std::cout << "    verify: copy back failed: "
                              << cudaGetErrorString(ce2) << std::endl;
                }
            }
        }
    }

    for (auto p : hA) if (p) cudaFree(p);
    for (auto p : hB) if (p) cudaFree(p);
    if (dA)    cudaFree(dA);
    if (dB)    cudaFree(dB);
    if (dIpiv) cudaFree(dIpiv);
    if (dInfo) cudaFree(dInfo);
    cublasDestroy(handle);

    return ok;
}

int main() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total memory: "
              << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB"
              << std::endl;
    std::cout << "Shared mem per block: " << prop.sharedMemPerBlock << " bytes"
              << std::endl;
    std::cout << "Shared mem per block optin: " << prop.sharedMemPerBlockOptin
              << " bytes" << std::endl;

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    int ver = 0;
    cublasGetVersion(handle, &ver);
    cublasDestroy(handle);
    std::cout << "cuBLAS version: " << ver << std::endl;
    std::cout << std::endl;

    std::vector<int> n_values;
    for (int n = 1024; n <= (1 << 20); n *= 2) n_values.push_back(n);

    const int nrhs = 1;
    const int batchSize = 1;

    std::cout << "########## float ##########" << std::endl;
    for (int n : n_values) {
        bool ok = test_getrs_batched<float>(n, nrhs, batchSize, true);
        if (!ok) {
            std::cout << "  >>> failed/skipped at n=" << n
                      << ", stopping float tests" << std::endl;
            break;
        }
    }

    std::cout << "\n########## double ##########" << std::endl;
    for (int n : n_values) {
        bool ok = test_getrs_batched<double>(n, nrhs, batchSize, true);
        if (!ok) {
            std::cout << "  >>> failed/skipped at n=" << n
                      << ", stopping double tests" << std::endl;
            break;
        }
    }

    return 0;
}