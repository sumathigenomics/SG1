from pathlib import Path

import cupy as cp
import numpy as np


MASK_2BIT = np.uint64(0x5555555555555555)


def sg1_mismatch_distance(block_a, block_b):
    """Return SG1 mismatch distance for two packed 64-bit DNA blocks."""
    diff = np.uint64(block_a) ^ np.uint64(block_b)
    norm = (diff | (diff >> np.uint64(1))) & MASK_2BIT
    return int(norm.bit_count())


def check_orthogonality_cpu(circuit_bits, query_bits, max_m=5):
    """Reference CPU implementation for validation and non-GPU environments."""
    circuit = np.asarray(circuit_bits, dtype=np.uint64)
    queries = np.asarray(query_bits, dtype=np.uint64)

    out = np.full((len(queries), len(circuit)), -1, dtype=np.int32)
    for q_idx, q in enumerate(queries):
        for c_idx, c in enumerate(circuit):
            dist = sg1_mismatch_distance(q, c)
            if dist <= max_m:
                out[q_idx, c_idx] = dist
    return out

class SG1Engine:
    def __init__(self, kernel_path="sg1_kernel.cu"):
        root = Path(__file__).resolve().parent
        candidate = Path(kernel_path)
        if not candidate.exists():
            candidate = root / kernel_path
        if not candidate.exists():
            raise FileNotFoundError(f"CUDA kernel file not found: {kernel_path}")

        with open(candidate, 'r', encoding='utf-8') as f:
            self.module = cp.RawModule(code=f.read())
        self.kernel = self.module.get_function("sg1_orthogonality_kernel")

    def check_orthogonality(self, circuit_bits, query_bits, max_m=5):
        num_c = len(circuit_bits)
        num_q = len(query_bits)
        
        # Move to GPU
        d_circuit = cp.asarray(circuit_bits, dtype=cp.uint64)
        d_queries = cp.asarray(query_bits, dtype=cp.uint64)
        d_results = cp.zeros((num_q, num_c), dtype=cp.int32)

        # Launch 2D Grid
        threads = (16, 16)
        grid = ((num_c + 15) // 16, (num_q + 15) // 16)

        self.kernel(grid, threads, (d_circuit, d_queries, d_results, num_c, num_q, max_m))

        return cp.asnumpy(d_results)

# Example SynBio Usage
# engine = SG1Engine()
# matrix = engine.check_orthogonality(my_plasmid_library, my_grnas)
