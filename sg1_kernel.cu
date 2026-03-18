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

        // The SG1 Transform: XOR -> OR-Shift -> Mask -> Popcount
        // XOR finds differing bits between query and component.
        // OR with right-shift collapses each 2-bit pair: if either bit differs, the LSB is set.
        // Mask isolates only the LSB of each 2-bit pair for counting.
        uint64_t diff = q ^ c;
        uint64_t norm = (diff | (diff >> 1)) & 0x5555555555555555ULL;
        int dist = __popcll(norm);

        // Store result: distance if within threshold, -1 if orthogonal (safe)
        output_matrix[q_idx * num_components + c_idx] = (dist <= max_mismatch) ? dist : -1;
    }
}
