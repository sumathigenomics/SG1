---

### **2. sg1_kernel.cu (The CUDA Core)**

```cpp
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__
void sg1_orthogonality_kernel(
    const uint64_t* circuit_library, 
    const uint64_t* grna_queries,
    int* output_matrix,
    int num_components,
    int num_queries,
    int max_mismatch) 
{
    // Map threads to the Orthogonality Matrix (Query x Component)
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx < num_queries && c_idx < num_components) {
        uint64_t q = grna_queries[q_idx];
        uint64_t c = circuit_library[c_idx];

        // The SG1 Transform: XOR -> Shift-OR -> Mask -> Popcount
        uint64_t diff = q ^ c;
        uint64_t norm = (diff | (diff >> 1)) & 0x5555555555555555ULL;
        int dist = __popcll(norm);

        // Store result in the 2D matrix
        output_matrix[q_idx * num_components + c_idx] = (dist <= max_mismatch) ? dist : -1;
    }
}
