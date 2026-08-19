#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t status_ = (call);                                                    \
        if (status_ != cudaSuccess) {                                                          \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status_)); \
        }                                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                           \
    do {                                                                             \
        const cublasStatus_t status_ = (call);                                        \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                       \
            throw std::runtime_error("cuBLAS error: " + std::to_string(status_));     \
        }                                                                            \
    } while (0)

struct TestCase {
    std::string name;
    std::string type;
    int n;
    int incx;
    int incy;
    std::size_t x_count;
    std::size_t y_count;
    std::string input_x;
    std::string input_y;
    std::string output_x;
    std::string output_y;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : pointer_(nullptr) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() { return pointer_; }

private:
    T *pointer_;
};

template <typename T>
std::vector<T> read_binary(const std::string &path, std::size_t count) {
    std::vector<T> data(count);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open input file: " + path);
    }

    const std::streamsize bytes = static_cast<std::streamsize>(count * sizeof(T));
    file.read(reinterpret_cast<char *>(data.data()), bytes);
    if (file.gcount() != bytes) {
        throw std::runtime_error("unexpected input size: " + path);
    }
    return data;
}

template <typename T>
void write_binary(const std::string &path, const std::vector<T> &data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot create output file: " + path);
    }

    file.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!file) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

bool indices_fit(std::size_t count, int n, int increment) {
    if (n <= 0) {
        return true;
    }

    const long long start = increment < 0
                                ? (1LL - static_cast<long long>(n)) * increment
                                : 0LL;
    const long long last = start + (static_cast<long long>(n) - 1LL) * increment;
    const long long minimum = start < last ? start : last;
    const long long maximum = start > last ? start : last;
    return minimum >= 0 && static_cast<unsigned long long>(maximum) < count;
}

cublasStatus_t call_swap(cublasHandle_t handle, int n, float *x, int incx,
                         float *y, int incy) {
    return cublasSswap(handle, n, x, incx, y, incy);
}

cublasStatus_t call_swap(cublasHandle_t handle, int n, double *x, int incx,
                         double *y, int incy) {
    return cublasDswap(handle, n, x, incx, y, incy);
}

cublasStatus_t call_swap(cublasHandle_t handle, int n, cuComplex *x, int incx,
                         cuComplex *y, int incy) {
    return cublasCswap(handle, n, x, incx, y, incy);
}

cublasStatus_t call_swap(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx,
                         cuDoubleComplex *y, int incy) {
    return cublasZswap(handle, n, x, incx, y, incy);
}

template <typename T>
void run_case(cublasHandle_t handle, const TestCase &test) {
    if (!indices_fit(test.x_count, test.n, test.incx) ||
        !indices_fit(test.y_count, test.n, test.incy)) {
        throw std::runtime_error("test would access outside its allocation: " + test.name);
    }

    std::vector<T> x = read_binary<T>(test.input_x, test.x_count);
    std::vector<T> y = read_binary<T>(test.input_y, test.y_count);
    DeviceBuffer<T> d_x(test.x_count);
    DeviceBuffer<T> d_y(test.y_count);

    CUDA_CHECK(cudaMemcpy(d_x.get(), x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y.get(), y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(call_swap(handle, test.n, d_x.get(), test.incx,
                           d_y.get(), test.incy));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(x.data(), d_x.get(), x.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(), d_y.get(), y.size() * sizeof(T), cudaMemcpyDeviceToHost));

    write_binary(test.output_x, x);
    write_binary(test.output_y, y);
}

std::vector<TestCase> read_manifest(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open manifest: " + path);
    }

    std::vector<TestCase> tests;
    TestCase test;
    while (file >> test.name >> test.type >> test.n >> test.incx >> test.incy >>
           test.x_count >> test.y_count >> test.input_x >> test.input_y >>
           test.output_x >> test.output_y) {
        tests.push_back(test);
    }

    if (!file.eof()) {
        throw std::runtime_error("malformed manifest: " + path);
    }
    return tests;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " manifest.txt\n";
        return EXIT_FAILURE;
    }

    cublasHandle_t handle = nullptr;
    try {
        const std::vector<TestCase> tests = read_manifest(argv[1]);
        CUBLAS_CHECK(cublasCreate(&handle));

        for (std::size_t index = 0; index < tests.size(); ++index) {
            const TestCase &test = tests[index];
            if (test.type == "S") {
                run_case<float>(handle, test);
            } else if (test.type == "D") {
                run_case<double>(handle, test);
            } else if (test.type == "C") {
                run_case<cuComplex>(handle, test);
            } else if (test.type == "Z") {
                run_case<cuDoubleComplex>(handle, test);
            } else {
                throw std::runtime_error("unknown data type: " + test.type);
            }
            std::cout << "[CUDA " << index + 1 << "/" << tests.size()
                      << "] " << test.name << '\n';
        }

        CUBLAS_CHECK(cublasDestroy(handle));
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
