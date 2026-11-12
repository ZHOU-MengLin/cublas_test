#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>


// nvcc swap_test.cu -lcublas -o swap_test
// ./swap_test

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

void print_vector(const char* name, const std::vector<double>& values)
{
    std::printf("%s = [", name);

    for (std::size_t i = 0; i < values.size(); ++i) {
        std::printf("%.1f%s",
                    values[i],
                    i + 1 == values.size() ? "" : ", ");
    }

    std::printf("]\n");
}



int main()
{
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    // std::vector<double> x = {1.0, 2.0, 3.0, 4.0,5.0,6.0,7.0,8.0};
    // std::vector<double> y = {5.0, 6.0, 7.0, 8.0};

    std::vector<double> x = {1.0, -INFINITY, 3.0, INFINITY,5.0,6.0,7.0,8.0};
    std::vector<double> y = {NAN, 6.0, 7.0, INFINITY};
    
    std::vector<double> y_storage = {
      5.0,   6.0,   7.0,   8.0,
    101.0, 102.0, 103.0, 104.0};
    int n = 4;
    int incx=2;
    int incy=1;
    std::cout << "n: ";
    std::cin >> n;

    std::cout << "incx: ";
    std::cin >> incx;

    std::cout << "incy: ";
    std::cin >> incy;
    
    double* d_x = nullptr;
    double* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_x),
        x.size() * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_y),
        y_storage.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(
        d_x,
        x.data(),
        x.size() * sizeof(double),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_y,
        y_storage.data(),
        y_storage.size() * sizeof(double),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_y,
        y.data(),
        y.size() * sizeof(double),
        cudaMemcpyHostToDevice));

    print_vector("x before", x);
    print_vector("y before", y);
    std::printf("n=%d, incx=%d, incy=%d\n", n, incx, incy);

    CUBLAS_CHECK(cublasDswap(
        handle,
        n,
        d_x,
        incx,
        d_y,
        incy));

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

    print_vector("x after ", x);
    print_vector("y after ", y);



    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));


    CUBLAS_CHECK(cublasDestroy(handle));

    return 1;
}