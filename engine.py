import cupy as cp
import numpy as np

class SG1Engine:
    def __init__(self, kernel_path="sg1_kernel.cu"):
        with open(kernel_path, 'r') as f:
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
