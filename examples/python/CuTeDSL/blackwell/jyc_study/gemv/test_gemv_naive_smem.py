import argparse
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

# S_(t-1) @ k_(t)^T: [128, 128] @ [128, 1]
mma_tiler_mnk = (128, 1, 128)
a_dtype = cutlass.Float32
b_dtype = cutlass.BFloat16
c_dtype = cutlass.Float32

"""
Below code gives a reference for Float32 GEMV (General Matrix-Vector Multiplication):

Given:
    - A: a matrix of shape (l, m, k), where l is the batch size * num heads, m is the number of rows, k is the number of columns. The data type is Float32
    - b: a batched vector of shape (l, k) and the data type is BFloat16.
    - c: the output batched vector of shape (l, m) and the data type is Float32.

Operation:
    c = A * b

Assumptions:
    - The matrix A is stored in memory such that the k (column) dimension is contiguous
    - The m dimension is a multiple of 128
    - The k dimension is a multiple of 64

"""


class Sm100DenseGemvKernelNaiveSmem:
    def __init__(self):
        self.threads_per_cta = 128

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # Compute grid size
        grid = (
            cute.ceil_div(c_tensor.shape[0], 128),
            1,
            c_tensor.shape[2],
        )
        # Launch the kernel synchronously
        self.kernel(a_tensor, b_tensor, c_tensor).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
    ):
        bidx, _, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        gA_offset = (
            bidz * gA.layout[cute.rank(gA.layout) - 1].stride
            + bidx * mma_tiler_mnk[0] * gA.layout[0].stride
            + tidx * gA.layout[0].stride
        )
        gB_offset = bidz * gB.layout[cute.rank(gB.layout) - 1].stride
        gC_offset = (
            bidz * gC.layout[2].stride
            + bidx * mma_tiler_mnk[0] * gC.layout[0].stride
            + tidx * gC.layout[0].stride
        )

        #
        # Each thread compute 1 results
        #
        tC = cute.make_tensor(gC.iterator + gC_offset, 1)
        c_res = cute.zeros_like(tC, cutlass.Float32)

        tAgA_ptr = gA.iterator + gA_offset
        tBgB_ptr = gB.iterator + gB_offset

        tAgA_ktile_offset = mma_tiler_mnk[2]
        tBgB_ktile_offset = mma_tiler_mnk[2]


        smem = cutlass.utils.SmemAllocator()
        sB = smem.allocate_tensor(b_dtype, cute.make_layout((mma_tiler_mnk[2],), stride=(1,)), 16)
        # cooperative load once per K-tile
        if tidx < mma_tiler_mnk[2]:
            tBsB = cute.make_tensor(sB.iterator + tidx, 1)
            tBgB = cute.make_tensor(tBgB_ptr + tidx, 1)
            tBgB_bf16 = tBgB.load()
            tBsB.store(tBgB_bf16)
        cute.arch.sync_threads()

        k_tile_cnt = cute.ceil_div(gA.layout[1].shape, mma_tiler_mnk[2])
        for k_tile in range(k_tile_cnt):
            # Create tensors for A/B tile
            tAgA = cute.make_tensor(tAgA_ptr, mma_tiler_mnk[2])
            a_val = tAgA.load()

            for i in cutlass.range_constexpr(mma_tiler_mnk[2]):
                b_val = sB[i].to(cutlass.Float32)
                c_res += (a_val[i] * b_val)
            # Update pointers for next tile
            tAgA_ptr += tAgA_ktile_offset
            tBgB_ptr += tBgB_ktile_offset

        # Store result to global memory
        tC.store(c_res)
        return


def run_gemv_naive_smem(
    m: int,
    k: int,
    l: int,
    tolerance: float,
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.
    """
    print("=" * 60)
    print("Launching Blackwell Float32 GEMV Naive + SMEM Test")
    print("-" * 60)
    print("Input dimensions:")
    print(f"  A: ({l}, {m}, {k}) [l: batch size * num heads, m: rows, k: cols]")
    print(f"  b: ({l}, {k}) [l: batch size * num heads, k: length]")
    print(f"  c: ({l}, {m}) [l: batch size * num heads, m: length]")
    print("Data types:")
    print(f"  A dtype: {a_dtype}")
    print(f"  b dtype: {b_dtype}")
    print(f"  Output c dtype: {c_dtype}")
    print(f"Validation tolerance: {tolerance}")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # GEMV, N must be 1
    n = 1

    # Create tensor A/B/C
    a_ref = cutlass_torch.matrix(l, m, k, False, cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, False, cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, True, cutlass.Float32)
    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref, a_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref, b_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    # Mark tensor with element divisibility for 16B alignment
    a_tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=32,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=32,
    )
    c_tensor.mark_compact_shape_dynamic(
        0,
        (2, 1, 0),
        divisibility=16,
    )

    # Configure gemv kernel
    gemv = Sm100DenseGemvKernelNaiveSmem()
    # Initialize Stream
    current_stream = cutlass_torch.default_stream()
    # Compile gemv kernel
    compiled_gemv = cute.compile(
        gemv,
        a_tensor,
        b_tensor,
        c_tensor,
        current_stream,
    )

    # Launch GPU kernel
    compiled_gemv(a_tensor, b_tensor, c_tensor, current_stream)

    # Compute reference result, simulate GEMV via 1 FFMA based matmul computations
    ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)

    # Convert c back to f32 for comparison.
    c_ref_device = c_ref.cuda()
    cute.testing.convert(
        c_tensor,
        from_dlpack(c_ref_device, assumed_align=16).mark_layout_dynamic(leading_dim=0),
    )
    c_ref = c_ref_device.cpu()
    torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example of Sm100 Dense Float32 GEMV."
    )
    parser.add_argument(
        "--m",
        type=int,
        default=128,
        help="m dimensions",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=128,
        help="k dimensions",
    )
    parser.add_argument(
        "--l",
        type=int,
        # default=[32*1],
        default=[32*1, 32*2, 32*4, 32*8, 32*16, 32*32, 32*64, 32*128, 32*256, 32*512],
        help="l dimension, batch size * num_heads (list of values)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Number of warmup iterations to run the kernel"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations to run the kernel"
    )
    parser.add_argument(
        "--evict-mb", type=int, default=128, help="optional L2/L1 eviction buffer size in MB (0 to disable)"
    )
    args = parser.parse_args()

    if args.k % mma_tiler_mnk[2] != 0:
        raise ValueError("K must be a multiple of 64 for this GEMV kernel.")
    if args.m % mma_tiler_mnk[0] != 0:
        raise ValueError("M must be a multiple of 128 for this GEMV kernel.")

    evict_buf = None
    if args.evict_mb > 0:
        elems = (args.evict_mb * 1024 * 1024) // 4  # float32
        evict_buf = torch.empty(elems, dtype=torch.float32, device="cuda")

    def evict_caches():
        if evict_buf is not None:
            evict_buf.add_(1.0)
            torch.cuda.synchronize()

    # warmup
    for i in range(args.warmup):
        for l in args.l:
            run_gemv_naive_smem(
                args.m,
                args.k,
                l,
                args.tolerance,
            )
            evict_caches()

    # benchmark
    for i in range(args.iterations):
        for l in args.l:
            run_gemv_naive_smem(
                args.m,
                args.k,
                l,
                args.tolerance,
            )
            evict_caches()
    print("PASS")
