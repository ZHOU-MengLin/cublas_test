#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------
// 1. GPU Kernel (设备端代码)
// ---------------------------
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

// ---------------------------
// 2. 结果检查函数 (Host端工具)
// ---------------------------
void checkResult(float *host_A, float *host_B, float *host_C, int numElements)
{
    double epsilon = 1.0E-5;
    bool match = true;

    printf("\n[Checking results...]\n");
    for (int i = 0; i < numElements; i++)
    {
        float expected = host_A[i] + host_B[i];
        if (fabs(expected - host_C[i]) > epsilon)
        {
            match = false;
            printf("Error at index %d: Expected %f, got %f\n", i, expected, host_C[i]);
            // 遇到错误可以打印前几个方便调试，这里直接 break
            break; 
        }
    }

    if (match)
        printf("Test PASSED: All %d elements match.\n", numElements);
    else
        printf("Test FAILED.\n");
}

// ---------------------------
// 3. Main 函数 (逻辑控制)
// ---------------------------
int main(void)
{
    // 定义规模
    int numElements = 1024*1024;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // --- Step 1: 申请内存 ---
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化数据
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // --- Step 2: H2D 数据拷贝 ---
    printf("Copying data from Host to Device...\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // --- Step 3: 启动 Kernel ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel...\n");
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // 同步 (对于调试很有用，确保 kernel 跑完了再继续)
    cudaDeviceSynchronize();

    // --- Step 4: D2H 数据拷贝 ---
    printf("Copying data from Device to Host...\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // --- Step 5: 验证结果 (调用新函数) ---
    checkResult(h_A, h_B, h_C, numElements);

    // --- Step 6: 释放内存 ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}