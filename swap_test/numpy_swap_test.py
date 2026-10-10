import argparse
import subprocess
from pathlib import Path

import numpy as np


def required_size(n, inc):
    if n <= 0:
        return 1
    return 1 + (n - 1) * abs(inc)


def vector_indices(n, inc):
    
    if n <= 0:
        return np.empty(0, dtype=np.int64)

    start = 0 if inc > 0 else (1 - n) * inc

    return start + np.arange(n, dtype=np.int64) * inc


def swap_golden(x, y, n, incx, incy):
    golden_x = x.copy()
    golden_y = y.copy()

    if n <= 0:
        return golden_x, golden_y

    if incx == 0 or incy == 0:
        raise ValueError("incx and incy cannot be zero")

    x_idx = vector_indices(n, incx)
    y_idx = vector_indices(n, incy)

    if x_idx.max() >= x.size or x_idx.min() < 0:
        raise ValueError(
            f"x storage is too small: size={x.size}, indices={x_idx}"
        )

    if y_idx.max() >= y.size or y_idx.min() < 0:
        raise ValueError(
            f"y storage is too small: size={y.size}, indices={y_idx}"
        )

    temporary = golden_x[x_idx].copy()
    golden_x[x_idx] = golden_y[y_idx]
    golden_y[y_idx] = temporary

    return golden_x, golden_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--incx", type=int, default=2)
    parser.add_argument("--incy", type=int, default=1)
    parser.add_argument("--x-size", type=int)
    parser.add_argument("--y-size", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--exe", default="./swap_test")
    args = parser.parse_args()

    if args.n < 0:
        raise ValueError("n cannot be negative")

    if args.n > 0 and (args.incx == 0 or args.incy == 0):
        raise ValueError("incx and incy cannot be zero")

    minimum_x_size = required_size(args.n, args.incx)
    minimum_y_size = required_size(args.n, args.incy)

    x_size = args.x_size or minimum_x_size
    y_size = args.y_size or minimum_y_size

    if x_size < minimum_x_size:
        raise ValueError(
            f"x-size must be at least {minimum_x_size}"
        )

    if y_size < minimum_y_size:
        raise ValueError(
            f"y-size must be at least {minimum_y_size}"
        )
    x_size = args.x_size
    y_size = args.y_size
    rng = np.random.default_rng(args.seed)

    # 使用整数生成 double 数据，输出更容易观察。
    x = rng.integers(1, 100, size=x_size).astype(np.float64)
    y = rng.integers(101, 200, size=y_size).astype(np.float64)

    golden_x, golden_y = swap_golden(
        x, y, args.n, args.incx, args.incy
    )

    x_idx = vector_indices(args.n, args.incx)
    y_idx = vector_indices(args.n, args.incy)

    # 原始二进制文件，C++ 可以直接读取。
    x.tofile("input_x.bin")
    y.tofile("input_y.bin")
    golden_x.tofile("golden_x.bin")
    golden_y.tofile("golden_y.bin")

    # 配置顺序：x_size y_size n incx incy
    Path("swap_config.txt").write_text(
        f"{x.size} {y.size} {args.n} "
        f"{args.incx} {args.incy}\n",
        encoding="utf-8",
    )

    print("===== Python generated data =====")
    print(f"n={args.n}, incx={args.incx}, incy={args.incy}")
    print(f"x indices: {x_idx}")
    print(f"y indices: {y_idx}")
    print(f"x before:  {x}")
    print(f"y before:  {y}")
    print(f"golden x:  {golden_x}")
    print(f"golden y:  {golden_y}")

    subprocess.run([args.exe], check=True)

    actual_x = np.fromfile("actual_x.bin", dtype=np.float64)
    actual_y = np.fromfile("actual_y.bin", dtype=np.float64)

    print("\n===== CUDA result =====")
    print(f"actual x:  {actual_x}")
    print(f"actual y:  {actual_y}")

    try:
        np.testing.assert_array_equal(actual_x, golden_x)
        np.testing.assert_array_equal(actual_y, golden_y)
    except AssertionError as error:
        print("\nResult: FAIL")
        print(error)
        raise SystemExit(1)

    print("\nResult: PASS")


if __name__ == "__main__":
    main()