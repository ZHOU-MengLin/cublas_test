#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
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

std::vector<double> read_binary(
    const std::string& filename,
    std::size_t size)
{
    std::vector<double> data(size);

    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot open " << filename << '\n';
        std::exit(EXIT_FAILURE);
    }

    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(size * sizeof(double)));

    if (!file) {
        std::cerr << "Failed to read " << filename << '\n';
        std::exit(EXIT_FAILURE);
    }

    return data;
}

void write_binary(
    const std::string& filename,
    const std::vector<double>& data)
{
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot create " << filename << '\n';
        std::exit(EXIT_FAILURE);
    }

    file.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(double)));

    if (!file) {
        std::cerr << "Failed to write " << filename << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::size_t required_size(int n, int inc)
{
    if (n <= 0) {
        return 1;
    }

    return 1 +
        static_cast<std::size_t>(n - 1) *
        static_cast<std::size_t>(std::abs(inc));
}

int main()
{
    std::size_t x_size = 0;
    std::size_t y_size = 0;
    int n = 0;
    int incx = 0;
    int incy = 0;

    std::ifstream config("swap_config.txt");

    if (!(config >> x_size >> y_size >> n >> incx >> incy)) {
        std::cerr << "Failed to read swap_config.txt\n";
        return EXIT_FAILURE;
    }

    if (n < 0) {
        std::cerr << "n cannot be negative\n";
        return EXIT_FAILURE;
    }

    if (n > 0 && (incx == 0 || incy == 0)) {
        std::cerr << "incx and incy cannot be zero\n";
        return EXIT_FAILURE;
    }

    if (x_size < required_size(n, incx) ||
        y_size < required_size(n, incy)) {
        std::cerr << "Input vector storage is too small\n";
        return EXIT_FAILURE;
    }

    std::vector<double> x = read_binary("input_x.bin", x_size);
    std::vector<double> y = read_binary("input_y.bin", y_size);

    std::cout
        << "CUDA: n=" << n
        << ", incx=" << incx
        << ", incy=" << incy
        << '\n';

    double* d_x = nullptr;
    double* d_y = nullptr;
    cublasHandle_t handle = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_x),
        x.size() * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_y),
        y.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(
        d_x,
        x.data(),
        x.size() * sizeof(double),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_y,
        y.data(),
        y.size() * sizeof(double),
        cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate(&handle));

    if (n > 0) {
        CUBLAS_CHECK(cublasDswap(
            handle,
            n,
            d_x,
            incx,
            d_y,
            incy));
    }

    CUDA_CHECK(cudaMemcpy(
        x.data(),
        d_x,
        x.size() * sizeof(double),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        y.data(),
        d_y,
        y.size() * sizeof(double),
        cudaMemcpyDeviceToHost));

    write_binary("actual_x.bin", x);
    write_binary("actual_y.bin", y);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return EXIT_SUCCESS;
}