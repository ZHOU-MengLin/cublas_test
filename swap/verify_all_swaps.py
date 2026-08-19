#!/usr/bin/env python3

import argparse
import random
import shutil
import subprocess
from pathlib import Path

import torch


TYPE_INFO = {
    "S": (torch.float32, "cublasSswap"),
    "D": (torch.float64, "cublasDswap"),
    "C": (torch.complex64, "cublasCswap"),
    "Z": (torch.complex128, "cublasZswap"),
}


TYPE_ALIASES = {
    "s": "S",
    "sswap": "S",
    "cublassswap": "S",
    "float32": "S",
    "d": "D",
    "dswap": "D",
    "cublasdswap": "D",
    "float64": "D",
    "c": "C",
    "cswap": "C",
    "cublascswap": "C",
    "complex64": "C",
    "z": "Z",
    "zswap": "Z",
    "cublaszswap": "Z",
    "complex128": "Z",
}


DETERMINISTIC_CASES = [
    ("n_negative", -1, 1, 1),
    ("n_zero", 0, 1, 1),
    ("single", 1, 1, 1),
    ("contiguous", 8, 1, 1),
    ("x_stride", 5, 2, 1),
    ("y_stride", 5, 1, 3),
    ("both_stride", 4, 2, 3),
    ("x_reverse", 6, -1, 1),
    ("y_reverse", 6, 1, -1),
    ("both_reverse", 6, -1, -1),
    ("reverse_stride", 4, -2, 3),
    ("x_zero_stride", 4, 0, 1),
    ("y_zero_stride", 4, 1, 0),
    ("both_zero_stride", 5, 0, 0),
]


def required_size(n, increment):
    if n <= 0:
        return 1
    return 1 + (n - 1) * abs(increment)


def blas_indices(n, increment):
    if n <= 0:
        return []
    start = (1 - n) * increment if increment < 0 else 0
    return [start + i * increment for i in range(n)]


def normalize_type(value):
    normalized = TYPE_ALIASES.get(value.lower())
    if normalized is None:
        supported = "S, D, C, Z, float32, float64, complex64, complex128"
        raise argparse.ArgumentTypeError(
            f"unsupported datatype {value!r}; supported values: {supported}"
        )
    return normalized


def validate_storage(name, count, n, increment):
    if count <= 0:
        raise ValueError(f"{name}-size must be greater than zero")
    indices = blas_indices(n, increment)
    if indices and (min(indices) < 0 or max(indices) >= count):
        raise ValueError(
            f"{name}-size={count} is too small: n={n}, inc{name[-1]}={increment}, "
            f"accessed indices={indices}"
        )


def swap_golden(x, y, n, incx, incy):
    """Sequential PyTorch Golden; repeated indices (increment=0) are intentional."""
    golden_x = x.clone()
    golden_y = y.clone()

    for x_index, y_index in zip(blas_indices(n, incx), blas_indices(n, incy)):
        temporary = golden_x[x_index].clone()
        golden_x[x_index] = golden_y[y_index]
        golden_y[y_index] = temporary

    return golden_x, golden_y


def random_tensor(count, dtype, generator):
    if dtype == torch.complex64:
        real = torch.randn(count, dtype=torch.float32, generator=generator)
        imag = torch.randn(count, dtype=torch.float32, generator=generator)
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(count, dtype=torch.float64, generator=generator)
        imag = torch.randn(count, dtype=torch.float64, generator=generator)
        return torch.complex(real, imag)
    return torch.randn(count, dtype=dtype, generator=generator)


def write_tensor(path, tensor):
    raw_bytes = tensor.contiguous().view(torch.uint8).tolist()
    path.write_bytes(bytes(raw_bytes))


def read_tensor(path, dtype, count):
    raw_bytes = bytearray(path.read_bytes())
    item_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = count * item_size
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"{path}: got {len(raw_bytes)} bytes, expected {expected_bytes}"
        )
    return torch.frombuffer(raw_bytes, dtype=dtype, count=count).clone()


def bitwise_equal(actual, expected):
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    return torch.equal(actual_bytes, expected_bytes)


def first_difference(actual, expected):
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    differing_bytes = torch.nonzero(actual_bytes != expected_bytes).flatten()
    if differing_bytes.numel() == 0:
        return "unknown difference"
    item_size = actual.element_size()
    index = int(differing_bytes[0].item() // item_size)
    return f"element {index}: actual={actual[index]}, expected={expected[index]}"


def build_cases(random_cases, seed):
    cases = list(DETERMINISTIC_CASES)
    rng = random.Random(seed)

    for index in range(random_cases):
        n = rng.randint(0, 32)
        incx = rng.randint(-4, 4)
        incy = rng.randint(-4, 4)
        cases.append((f"random_{index:03d}", n, incx, incy))
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch Goldens and validate Sswap/Dswap/Cswap/Zswap"
    )
    parser.add_argument("--runner", default="./swap_batch_runner")
    parser.add_argument("--random-cases", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--work-dir", default="swap_validation_data")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--x-size", type=int)
    parser.add_argument("--y-size", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--incx", type=int)
    parser.add_argument("--incy", type=int)
    parser.add_argument(
        "--datatype",
        "--datateype",
        dest="datatype",
        type=normalize_type,
        help="S/D/C/Z or float32/float64/complex64/complex128",
    )
    args = parser.parse_args()

    if args.random_cases < 0:
        raise ValueError("--random-cases cannot be negative")

    single_values = {
        "--x-size": args.x_size,
        "--y-size": args.y_size,
        "--n": args.n,
        "--incx": args.incx,
        "--incy": args.incy,
        "--datateype": args.datatype,
    }
    single_mode = any(value is not None for value in single_values.values())
    if single_mode:
        missing = [name for name, value in single_values.items() if value is None]
        if missing:
            parser.error(
                "single-case mode requires all parameters; missing: " + ", ".join(missing)
            )
        validate_storage("x", args.x_size, args.n, args.incx)
        validate_storage("y", args.y_size, args.n, args.incy)

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    if single_mode:
        cases = [("custom", args.n, args.incx, args.incy, args.x_size, args.y_size)]
        selected_types = {args.datatype: TYPE_INFO[args.datatype]}
        print(
            "Single case: "
            f"datatype={args.datatype}, x_size={args.x_size}, y_size={args.y_size}, "
            f"n={args.n}, incx={args.incx}, incy={args.incy}"
        )
    else:
        cases = [
            (name, n, incx, incy, required_size(n, incx) + 3,
             required_size(n, incy) + 2)
            for name, n, incx, incy in build_cases(args.random_cases, args.seed)
        ]
        selected_types = TYPE_INFO

    records = []
    manifest_lines = []
    case_number = 0

    for type_code, (torch_dtype, routine) in selected_types.items():
        for case_name, n, incx, incy, x_count, y_count in cases:
            case_number += 1
            full_name = f"{type_code}_{case_name}"
            x = random_tensor(x_count, torch_dtype, generator)
            y = random_tensor(y_count, torch_dtype, generator)
            golden_x, golden_y = swap_golden(x, y, n, incx, incy)

            input_x = work_dir / f"{case_number:04d}_input_x.bin"
            input_y = work_dir / f"{case_number:04d}_input_y.bin"
            output_x = work_dir / f"{case_number:04d}_output_x.bin"
            output_y = work_dir / f"{case_number:04d}_output_y.bin"
            golden_x_path = work_dir / f"{case_number:04d}_golden_x.bin"
            golden_y_path = work_dir / f"{case_number:04d}_golden_y.bin"
            write_tensor(input_x, x)
            write_tensor(input_y, y)
            write_tensor(golden_x_path, golden_x)
            write_tensor(golden_y_path, golden_y)

            manifest_lines.append(
                f"{full_name} {type_code} {n} {incx} {incy} "
                f"{x_count} {y_count} {input_x} {input_y} {output_x} {output_y}"
            )
            records.append(
                (full_name, routine, torch_dtype, x_count, y_count,
                 output_x, output_y, golden_x, golden_y)
            )

    manifest = work_dir / "manifest.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(f"Running {len(records)} cases in one CUDA process...")
    subprocess.run([args.runner, str(manifest)], check=True)

    passed_by_type = {type_code: 0 for type_code in selected_types}
    failures = []
    for record in records:
        (name, routine, torch_dtype, x_count, y_count,
         output_x, output_y, golden_x, golden_y) = record
        actual_x = read_tensor(output_x, torch_dtype, x_count)
        actual_y = read_tensor(output_y, torch_dtype, y_count)

        x_ok = bitwise_equal(actual_x, golden_x)
        y_ok = bitwise_equal(actual_y, golden_y)
        if x_ok and y_ok:
            passed_by_type[name[0]] += 1
        else:
            details = []
            if not x_ok:
                details.append("x " + first_difference(actual_x, golden_x))
            if not y_ok:
                details.append("y " + first_difference(actual_y, golden_y))
            failures.append(f"{name} ({routine}): " + "; ".join(details))

    print("\n===== Validation summary =====")
    per_type = len(cases)
    for type_code, (_, routine) in selected_types.items():
        print(f"{routine:12s}: {passed_by_type[type_code]}/{per_type} passed")
    print(f"Total       : {len(records) - len(failures)}/{len(records)} passed")

    if failures:
        print("\n===== Failures =====")
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    if not args.keep_data:
        shutil.rmtree(work_dir)
    print("Result      : PASS")


if __name__ == "__main__":
    main()
