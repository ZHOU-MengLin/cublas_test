# Standalone cuBLAS swap validation

This directory validates `cublasSswap`, `cublasDswap`, `cublasCswap`, and
`cublasZswap` without depending on other files in CUDA Library Samples.
Python only requires PyTorch; NumPy is not used.

The flow is:

1. Python/PyTorch generates input tensors and sequential Golden results.
2. One CUDA process executes every test case through cuBLAS.
3. Python reads the full output allocations and compares them bit for bit.

Build and run on a Linux CUDA server:

```bash
nvcc -std=c++11 -lineinfo swap_batch_runner.cu -lcublas -o swap_batch_runner
python3 verify_all_swaps.py --random-cases 25
```

Run one explicitly configured case (`--datateype` is accepted as requested;
`--datatype` is an equivalent spelling):

```bash
python3 verify_all_swaps.py \
  --x-size 8 --y-size 4 \
  --n 4 --incx 2 --incy 1 \
  --datateype D
```

Accepted datatype values are `S`, `D`, `C`, `Z`, `float32`, `float64`,
`complex64`, and `complex128`. All six single-case arguments are required
together. The program rejects configurations whose accessed indices exceed the
specified allocation sizes.

Preserve generated inputs, outputs, Goldens, and the manifest for diagnosis:

```bash
python3 verify_all_swaps.py --random-cases 100 --keep-data
```

Deterministic coverage includes negative/zero length, scalar, contiguous,
positive strides, negative strides, mixed directions, and repeated-address zero strides.
Random coverage selects `n` from 0 through 32 and increments from -4 through 4.
Every generated case is allocation-safe; out-of-bounds behavior is deliberately
excluded because it has no valid Golden result.
