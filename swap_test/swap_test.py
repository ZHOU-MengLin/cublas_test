import argparse
import torch


def blas_indices(n: int, inc: int):
    """
    生成符合当前 cuBLAS 实测行为的访问索引。

    n=3, inc= 1 -> [0, 1, 2]
    n=3, inc=-1 -> [2, 1, 0]
    n=3, inc= 2 -> [0, 2, 4]
    n=3, inc=-2 -> [4, 2, 0]
    n=3, inc= 0 -> [0, 0, 0]
    """
    if n <= 0:
        return []

    start = (1 - n) * inc if inc < 0 else 0

    return [start + i * inc for i in range(n)]


def swap_golden(
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    incx: int,
    incy: int,
):
    golden_x = x.clone()
    golden_y = y.clone()

    # cuBLAS实测：n=0或负数时不修改输入
    if n <= 0:
        return golden_x, golden_y

    x_indices = blas_indices(n, incx)
    y_indices = blas_indices(n, incy)

    # 防止测试程序访问越界
    if min(x_indices) < 0 or max(x_indices) >= x.numel():
        raise IndexError(
            f"x越界：x.size={x.numel()}, 访问索引={x_indices}"
        )

    if min(y_indices) < 0 or max(y_indices) >= y.numel():
        raise IndexError(
            f"y越界：y.size={y.numel()}, 访问索引={y_indices}"
        )

    # 必须逐对交换。
    # inc=0时同一元素会被重复访问，交换顺序会影响最终结果。
    for x_index, y_index in zip(x_indices, y_indices):
        temporary = golden_x[x_index].clone()
        golden_x[x_index] = golden_y[y_index]
        golden_y[y_index] = temporary

    return golden_x, golden_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--incx", type=int, required=True)
    parser.add_argument("--incy", type=int, required=True)
    args = parser.parse_args()

    x = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        dtype=torch.float64,
    )
    

    y = torch.tensor(
        [5.0, 6.0, 7.0, 8.0],
        dtype=torch.float64,
    )

    golden_x, golden_y = swap_golden(
        x,
        y,
        args.n,
        args.incx,
        args.incy,
    )

    print(f"n={args.n}, incx={args.incx}, incy={args.incy}")
    print(f"x indices = {blas_indices(args.n, args.incx)}")
    print(f"y indices = {blas_indices(args.n, args.incy)}")
    print(f"x before  = {x.tolist()}")
    print(f"y before  = {y.tolist()}")
    print(f"golden x  = {golden_x.tolist()}")
    print(f"golden y  = {golden_y.tolist()}")


if __name__ == "__main__":
    main()