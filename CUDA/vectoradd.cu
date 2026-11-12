#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 检查 CUDA 运行时错误的宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ---- 修复后的 Kernel 函数 ----
__global__ void vector_add_half_optimized(const half *A, const half *B, half *C, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 策略：使用 float4 加载 128 bits (即 8 个 half)
    // 假设 numElements 是 8 的倍数且地址对齐
    const float4* A_vec = reinterpret_cast<const float4*>(A);
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    float4* C_vec = reinterpret_cast<float4*>(C);

    for (int i = idx; i < numElements / 8; i += stride) {
        // 1. 向量化加载 128 bits 数据
        float4 a_raw = A_vec[i];
        float4 b_raw = B_vec[i];
        float4 c_raw;

        // 2. 将 float4 的成员 (32 bits) 解释为 half2
        half2* a_h2 = reinterpret_cast<half2*>(&a_raw);
        half2* b_h2 = reinterpret_cast<half2*>(&b_raw);
        half2* c_h2 = reinterpret_cast<half2*>(&c_raw);

        // 3. 执行 SIMD 计算 (Unrolled loop for 4 half2 pairs)
        // [修复点] 增加数组索引，处理 4 个 half2 对
        c_h2[0] = __hadd2(a_h2[0], b_h2[0]); // 处理第 0,1 个 half
        c_h2[1] = __hadd2(a_h2[1], b_h2[1]); // 处理第 2,3 个 half
        c_h2[2] = __hadd2(a_h2[2], b_h2[2]); // 处理第 4,5 个 half
        c_h2[3] = __hadd2(a_h2[3], b_h2[3]); // 处理第 6,7 个 half

        // 4. 向量化存储
        C_vec[i] = c_raw;
    }

    // 处理剩余元素 (Peeling loop for remaining < 8 elements)
    int remaining_start = (numElements / 8) * 8;
    for (int i = remaining_start + idx; i < numElements; i += stride) {
        // 使用 intrinsics 标量加法
        C[i] = __hadd(A[i], B[i]);
    }
}

// ---- 补充的 Main 函数 ----
int main() {
    // 设置一个非 8 的整数倍，以验证剩余元素的处理逻辑是否正确
    int numElements = 1024 * 1024 + 5; 
    size_t size = numElements * sizeof(half);

    // 1. 分配 Host 内存
    std::vector<half> h_A(numElements);
    std::vector<half> h_B(numElements);
    std::vector<half> h_C(numElements);

    // 2. 初始化 Host 数据 (使用 __float2half 进行 CPU 端转换)
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = __float2half(1.5f);
        h_B[i] = __float2half(2.5f);
    }

    // 3. 分配 Device 内存
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // 4. 拷贝数据到 Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // 5. 配置 Kernel 启动参数
    int threadsPerBlock = 256;
    // 因为每个线程处理 8 个元素，所以需要的总逻辑线程数是 numElements / 8
    int vectorElements = numElements / 8;
    int blocksPerGrid = (vectorElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // 限制最大 Block 数量，利用 Grid-Stride 机制处理超出的工作负载
    if (blocksPerGrid > 1024) blocksPerGrid = 1024; 

    // 6. 启动 Kernel
    vector_add_half_optimized<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // 检查 Kernel 启动是否报错
    CUDA_CHECK(cudaGetLastError());
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. 将结果拷回 Host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // 8. 验证结果
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        // 使用 __half2float 将结果转回 float 进行精度校验
        float val = __half2float(h_C[i]);
        if (std::abs(val - 4.0f) > 1e-4) {
            std::cerr << "Mismatch at index " << i << ": expected 4.0, got " << val << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Success! All " << numElements << " elements computed correctly." << std::endl;
    }

    // 9. 释放资源
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}