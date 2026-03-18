import numpy as np

# CuPy (GPU) is optional — fall back to CPU if unavailable
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class SG1Engine:
    """
    SG1 SENTINEL Engine
    Performs bit-packed 2-bit DNA orthogonality checks.
    Uses GPU (CuPy/CUDA) if available, otherwise falls back to CPU (NumPy).
    """

    def __init__(self, kernel_path="sg1_kernel.cu"):
        self.use_gpu = GPU_AVAILABLE
        if self.use_gpu:
            try:
                with open(kernel_path, 'r') as f:
                    source = f.read()
                self.module = cp.RawModule(code=source)
                self.kernel = self.module.get_function("sg1_orthogonality_kernel")
                print("[SG1] GPU mode active.")
            except Exception as e:
                print(f"[SG1] GPU init failed ({e}). Falling back to CPU.")
                self.use_gpu = False
        else:
            print("[SG1] CuPy not found. Running in CPU mode.")

    def _cpu_check(self, circuit_bits, query_bits, max_m):
        """Pure NumPy fallback for the SG1 Transform."""
        num_q = len(query_bits)
        num_c = len(circuit_bits)
        results = np.empty((num_q, num_c), dtype=np.int32)

        for i, q in enumerate(query_bits):
            diff = circuit_bits ^ q
            norm = (diff | (diff >> np.uint64(1))) & np.uint64(0x5555555555555555)
            # Count set bits (popcount) per element
            dist = np.array([bin(int(x)).count('1') for x in norm], dtype=np.int32)
            results[i] = np.where(dist <= max_m, dist, -1)

        return results

    def check_orthogonality(self, circuit_bits, query_bits, max_m=5):
        """
        Run orthogonality check.

        Args:
            circuit_bits: array-like of uint64, the circuit component library
            query_bits:   array-like of uint64, the gRNA query sequences
            max_m:        int, mismatch threshold (default 5)

        Returns:
            NumPy int32 matrix (num_queries x num_components).
            Values 0..max_m = potential cross-talk distance.
            Value -1 = orthogonal (safe).
        """
        circuit_bits = np.asarray(circuit_bits, dtype=np.uint64)
        query_bits = np.asarray(query_bits, dtype=np.uint64)

        if self.use_gpu:
            num_c = len(circuit_bits)
            num_q = len(query_bits)

            d_circuit = cp.asarray(circuit_bits)
            d_queries = cp.asarray(query_bits)
            d_results = cp.zeros((num_q, num_c), dtype=cp.int32)

            threads = (16, 16)
            grid = ((num_c + 15) // 16, (num_q + 15) // 16)

            self.kernel(grid, threads, (d_circuit, d_queries, d_results,
                                        np.int32(num_c), np.int32(num_q), np.int32(max_m)))
            return cp.asnumpy(d_results)
        else:
            return self._cpu_check(circuit_bits, query_bits, max_m)
