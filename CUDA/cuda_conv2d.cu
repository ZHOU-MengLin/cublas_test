#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

template <
    typename InT,
    typename WeightT,
    typename OutT,
    int TILE_CO,
    int TILE_OW
>
__global__ void conv2d_cuda_core_fallback_kernel(
    const InT* __restrict__ X,
    const WeightT* __restrict__ W,
    const OutT* __restrict__ B,
    OutT* __restrict__ Y,
    int N, int Ci, int Hi, int Wi,
    int Co, int Ho, int Wo,
    int Kh, int Kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int has_bias
) {
    int ow_local = threadIdx.x;      // 0 .. TILE_OW-1
    int co_g     = threadIdx.y;      // 0 .. TILE_CO/4-1

    int ow = blockIdx.x * TILE_OW + ow_local;
    int ho = blockIdx.y;

    int num_co_blk = (Co + TILE_CO - 1) / TILE_CO;
    int z = blockIdx.z;
    int co_blk_id = z % num_co_blk;
    int n = z / num_co_blk;

    int co0 = co_blk_id * TILE_CO + co_g * 4;

    if (n >= N || ho >= Ho || ow >= Wo) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    if (has_bias) {
        if (co0 + 0 < Co) acc0 = static_cast<float>(B[co0 + 0]);
        if (co0 + 1 < Co) acc1 = static_cast<float>(B[co0 + 1]);
        if (co0 + 2 < Co) acc2 = static_cast<float>(B[co0 + 2]);
        if (co0 + 3 < Co) acc3 = static_cast<float>(B[co0 + 3]);
    }

    for (int ci = 0; ci < Ci; ++ci) {
        for (int kh = 0; kh < Kh; ++kh) {
            int hi = ho * stride_h - pad_h + kh * dilation_h;
            if (hi < 0 || hi >= Hi) continue;

            for (int kw = 0; kw < Kw; ++kw) {
                int wi = ow * stride_w - pad_w + kw * dilation_w;
                if (wi < 0 || wi >= Wi) continue;

                int x_idx = ((n * Ci + ci) * Hi + hi) * Wi + wi;
                float x = static_cast<float>(X[x_idx]);

                if (co0 + 0 < Co) {
                    int w_idx0 = (((co0 + 0) * Ci + ci) * Kh + kh) * Kw + kw;
                    acc0 += x * static_cast<float>(W[w_idx0]);
                }

                if (co0 + 1 < Co) {
                    int w_idx1 = (((co0 + 1) * Ci + ci) * Kh + kh) * Kw + kw;
                    acc1 += x * static_cast<float>(W[w_idx1]);
                }

                if (co0 + 2 < Co) {
                    int w_idx2 = (((co0 + 2) * Ci + ci) * Kh + kh) * Kw + kw;
                    acc2 += x * static_cast<float>(W[w_idx2]);
                }

                if (co0 + 3 < Co) {
                    int w_idx3 = (((co0 + 3) * Ci + ci) * Kh + kh) * Kw + kw;
                    acc3 += x * static_cast<float>(W[w_idx3]);
                }
            }
        }
    }

    if (co0 + 0 < Co) {
        int y_idx0 = ((n * Co + co0 + 0) * Ho + ho) * Wo + ow;
        Y[y_idx0] = static_cast<OutT>(acc0);
    }

    if (co0 + 1 < Co) {
        int y_idx1 = ((n * Co + co0 + 1) * Ho + ho) * Wo + ow;
        Y[y_idx1] = static_cast<OutT>(acc1);
    }

    if (co0 + 2 < Co) {
        int y_idx2 = ((n * Co + co0 + 2) * Ho + ho) * Wo + ow;
        Y[y_idx2] = static_cast<OutT>(acc2);
    }

    if (co0 + 3 < Co) {
        int y_idx3 = ((n * Co + co0 + 3) * Ho + ho) * Wo + ow;
        Y[y_idx3] = static_cast<OutT>(acc3);
    }
}

static int calc_out_size(
    int input,
    int kernel,
    int stride,
    int pad,
    int dilation
) {
    return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

void conv2d_cpu_reference(
    const std::vector<float>& X,
    const std::vector<float>& W,
    const std::vector<float>& B,
    std::vector<float>& Y,
    int N, int Ci, int Hi, int Wi,
    int Co, int Ho, int Wo,
    int Kh, int Kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int has_bias
) {
    std::fill(Y.begin(), Y.end(), 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Co; ++co) {
            for (int ho = 0; ho < Ho; ++ho) {
                for (int ow = 0; ow < Wo; ++ow) {
                    float acc = 0.0f;

                    if (has_bias) {
                        acc = B[co];
                    }

                    for (int ci = 0; ci < Ci; ++ci) {
                        for (int kh = 0; kh < Kh; ++kh) {
                            int hi = ho * stride_h - pad_h + kh * dilation_h;
                            if (hi < 0 || hi >= Hi) continue;

                            for (int kw = 0; kw < Kw; ++kw) {
                                int wi = ow * stride_w - pad_w + kw * dilation_w;
                                if (wi < 0 || wi >= Wi) continue;

                                int x_idx = ((n * Ci + ci) * Hi + hi) * Wi + wi;
                                int w_idx = (((co * Ci + ci) * Kh + kh) * Kw + kw);

                                acc += X[x_idx] * W[w_idx];
                            }
                        }
                    }

                    int y_idx = ((n * Co + co) * Ho + ho) * Wo + ow;
                    Y[y_idx] = acc;
                }
            }
        }
    }
}

void fill_random(std::vector<float>& v, float low = -1.0f, float high = 1.0f) {
    static std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(low, high);

    for (auto& x : v) {
        x = dist(gen);
    }
}

bool compare_result(
    const std::vector<float>& ref,
    const std::vector<float>& got,
    float atol,
    float rtol,
    int max_print = 10
) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int bad_count = 0;
    int first_bad = -1;

    for (int i = 0; i < static_cast<int>(ref.size()); ++i) {
        float abs_err = std::fabs(ref[i] - got[i]);
        float rel_err = abs_err / (std::fabs(ref[i]) + 1e-12f);

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);

        bool ok = abs_err <= atol + rtol * std::fabs(ref[i]);

        if (!ok) {
            if (first_bad < 0) {
                first_bad = i;
            }

            if (bad_count < max_print) {
                std::cout << "Mismatch [" << i << "] "
                          << "ref=" << std::setprecision(8) << ref[i]
                          << ", got=" << got[i]
                          << ", abs_err=" << abs_err
                          << ", rel_err=" << rel_err
                          << std::endl;
            }

            bad_count++;
        }
    }

    std::cout << "max_abs_err = " << max_abs_err
              << ", max_rel_err = " << max_rel_err
              << ", bad_count = " << bad_count
              << std::endl;

    return bad_count == 0;
}

template <int TILE_CO, int TILE_OW>
bool run_one_case(
    int case_id,
    int N, int Ci, int Hi, int Wi,
    int Co,
    int Kh, int Kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int has_bias
) {
    int Ho = calc_out_size(Hi, Kh, stride_h, pad_h, dilation_h);
    int Wo = calc_out_size(Wi, Kw, stride_w, pad_w, dilation_w);

    if (Ho <= 0 || Wo <= 0) {
        std::cout << "[Case " << case_id << "] Skip invalid output size."
                  << std::endl;
        return true;
    }

    size_t x_size = static_cast<size_t>(N) * Ci * Hi * Wi;
    size_t w_size = static_cast<size_t>(Co) * Ci * Kh * Kw;
    size_t b_size = static_cast<size_t>(Co);
    size_t y_size = static_cast<size_t>(N) * Co * Ho * Wo;

    std::vector<float> h_X(x_size);
    std::vector<float> h_W(w_size);
    std::vector<float> h_B(b_size);
    std::vector<float> h_Y_cpu(y_size);
    std::vector<float> h_Y_gpu(y_size, 0.0f);

    fill_random(h_X, -1.0f, 1.0f);
    fill_random(h_W, -1.0f, 1.0f);
    fill_random(h_B, -0.5f, 0.5f);

    conv2d_cpu_reference(
        h_X, h_W, h_B, h_Y_cpu,
        N, Ci, Hi, Wi,
        Co, Ho, Wo,
        Kh, Kw,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        has_bias
    );

    float* d_X = nullptr;
    float* d_W = nullptr;
    float* d_B = nullptr;
    float* d_Y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_X, x_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, y_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), x_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Y, 0, y_size * sizeof(float)));

    int num_co_blk = (Co + TILE_CO - 1) / TILE_CO;

    dim3 block(TILE_OW, TILE_CO / 4, 1);
    dim3 grid(
        (Wo + TILE_OW - 1) / TILE_OW,
        Ho,
        N * num_co_blk
    );

    conv2d_cuda_core_fallback_kernel<
        float,
        float,
        float,
        TILE_CO,
        TILE_OW
    ><<<grid, block>>>(
        d_X,
        d_W,
        d_B,
        d_Y,
        N, Ci, Hi, Wi,
        Co, Ho, Wo,
        Kh, Kw,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        has_bias
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_Y_gpu.data(), d_Y, y_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "============================================================" << std::endl;
    std::cout << "[Case " << case_id << "]" << std::endl;
    std::cout << "N=" << N
              << ", Ci=" << Ci
              << ", Hi=" << Hi
              << ", Wi=" << Wi
              << ", Co=" << Co
              << ", Ho=" << Ho
              << ", Wo=" << Wo
              << ", Kh=" << Kh
              << ", Kw=" << Kw
              << ", stride=(" << stride_h << "," << stride_w << ")"
              << ", pad=(" << pad_h << "," << pad_w << ")"
              << ", dilation=(" << dilation_h << "," << dilation_w << ")"
              << ", bias=" << has_bias
              << std::endl;

    std::cout << "TILE_CO=" << TILE_CO
              << ", TILE_OW=" << TILE_OW
              << ", grid=(" << grid.x << "," << grid.y << "," << grid.z << ")"
              << ", block=(" << block.x << "," << block.y << "," << block.z << ")"
              << std::endl;

    bool pass = compare_result(h_Y_cpu, h_Y_gpu, 1e-4f, 1e-4f);

    std::cout << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Y));

    return pass;
}

int main() {
    constexpr int TILE_CO = 16;
    constexpr int TILE_OW = 8;

    static_assert(TILE_CO % 4 == 0, "TILE_CO must be multiple of 4");

    bool all_pass = true;
    int case_id = 0;

    // case 0: 最普通的 3x3 conv，Co 是 TILE_CO 整数倍，Wo 是 TILE_OW 整数倍
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        1, 3, 8, 8,
        16,
        3, 3,
        1, 1,
        1, 1,
        1, 1,
        1
    );

    // case 1: Co 不是 4 的倍数，验证 co0 + 1/2/3 边界
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        1, 3, 8, 8,
        10,
        3, 3,
        1, 1,
        1, 1,
        1, 1,
        1
    );

    // case 2: Co 不是 TILE_CO 的倍数
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        2, 3, 9, 11,
        23,
        3, 3,
        1, 1,
        1, 1,
        1, 1,
        1
    );

    // case 3: Wo 不是 TILE_OW 的倍数，验证 ow 越界保护
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        1, 3, 7, 13,
        16,
        3, 3,
        1, 1,
        1, 1,
        1, 1,
        1
    );

    // case 4: 无 bias
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        1, 4, 10, 10,
        16,
        3, 3,
        1, 1,
        1, 1,
        1, 1,
        0
    );

    // case 5: stride = 2
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        1, 3, 15, 17,
        20,
        3, 3,
        2, 2,
        1, 1,
        1, 1,
        1
    );

    // case 6: 非对称 stride_h / stride_w
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        1, 3, 15, 17,
        20,
        3, 3,
        1, 2,
        1, 1,
        1, 1,
        1
    );

    // case 7: pad_h / pad_w 不同
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        1, 3, 12, 13,
        18,
        3, 5,
        1, 1,
        1, 2,
        1, 1,
        1
    );

    // case 8: dilation = 2
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        1, 3, 16, 16,
        17,
        3, 3,
        1, 1,
        2, 2,
        2, 2,
        1
    );

    // case 9: 1x1 conv
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        2, 5, 8, 8,
        13,
        1, 1,
        1, 1,
        0, 0,
        1, 1,
        1
    );

    // case 10: 大一点的 case
    all_pass &= run_one_case<TILE_CO, TILE_OW>(
        case_id++,
        4, 8, 32, 35,
        31,
        3, 3,
        1, 1,
        1, 1,
        1, 1,
        1
    );

    std::cout << "============================================================" << std::endl;

    if (all_pass) {
        std::cout << "ALL TESTS PASSED." << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED." << std::endl;
        return 1;
    }
}