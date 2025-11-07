import argparse
import torch

import cutlass
import torch.cuda.nvtx as nvtx

# S_(t-1) @ k_(t)^T: [128, 128] @ [128, 1]
mma_tiler_mnk = (128, 1, 128)

from test_gemv_naive import run_gemv_naive
from test_gemv_naive_smem import run_gemv_naive_smem
# from test_gemv_naive_smem_tmaload import run_gemv_naive_smem_tmaload
from test_gemv_naive_smem_bank import run_gemv_naive_smem_bank

from test_gemv_cute_layout import run_gemv_cute_layout


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
        # default=[32*128],
        default=[32*1, 32*2, 32*4, 32*8, 32*16, 32*32, 32*64, 32*128, 32*256, 32*512],
        # default=[32*1, 32*4, 32*16, 32*64, 32*128, 32*256],
        # default=[32*128, 32*256],
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
    print(f"Warmup {args.warmup} iterations")
    for i in range(args.warmup):
        with torch.cuda.nvtx.range(f"warmup_{i}"):
            for l in args.l:
                run_gemv_naive(
                    args.m,
                    args.k,
                    l,
                    args.tolerance,
                )
                run_gemv_naive_smem(
                    args.m,
                    args.k,
                    l,
                    args.tolerance,
                )
                # run_gemv_naive_smem_tmaload(
                #     args.m,
                #     args.k,
                #     l,
                #     args.tolerance,
                # )
                run_gemv_naive_smem_bank(
                    args.m,
                    args.k,
                    l,
                    args.tolerance,
                )
                # run_gemv_cute_layout(
                #     args.m,
                #     args.k,
                #     l,
                #     args.tolerance,
                # )
    evict_caches()

    # benchmark
    print(f"Benchmark {args.iterations} iterations")
    for i in range(args.iterations):
        with torch.cuda.nvtx.range(f"benchmark_{i}"):
            for l in args.l:
                run_gemv_naive(
                    args.m,
                    args.k,
                    l,
                    args.tolerance,
                )
                evict_caches()
                run_gemv_naive_smem(
                    args.m,
                    args.k,
                    l,
                    args.tolerance,
                )
                evict_caches()
                # run_gemv_naive_smem_tmaload(
                #     args.m,
                #     args.k,
                #     l,
                #     args.tolerance,
                # )
                # evict_caches()
                run_gemv_naive_smem_bank(
                    args.m,
                    args.k,
                    l,
                    args.tolerance,
                )
                evict_caches()
                # run_gemv_cute_layout(
                #     args.m,
                #     args.k,
                #     l,
                #     args.tolerance,
                # )
                # evict_caches()

    print("PASS")