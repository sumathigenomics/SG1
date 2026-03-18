# SG1 SENTINEL
High-speed bit-packed CUDA engine for genetic circuit orthogonality.
# SG1: SENTINEL High-Speed Bit-Packed CUDA Engine for Synthetic Biology

**Developed by:** Sumathi Genomics Co., Ltd. (Thailand) & Saminathan Industries Pte. Ltd. (Singapore)  
**Lead Researcher:** Dr. Saminathan Sivaprakasham Murugesan (saminathan@u.nus.edu)

## Overview
SG1 SENTINEL is a hardware-accelerated "Biological Design Rule Checker" (DRC). It utilizes 2-bit DNA vectorization and NVIDIA CUDA kernels to perform ultra-fast orthogonality verification and cross-talk analysis for synthetic genetic circuits.

## Key Features
* **Bit-Packed DNA Encoding:** Reduces genomic footprint by 75%, enabling entire host genomes to reside in GPU L2 cache.
* **SG1 Mismatch Transform:** A proprietary bitwise logic that normalizes 2-bit Hamming distances in a single clock cycle.
* **Orthogonality Matrix Mode:** All-vs-all comparison of gRNA regulators against synthetic circuit components.
* **Sub-second Latency:** Full human genome (GRCh38) scans in <800ms.

## Core Logic (The SG1 Transform)
For each query-component pair, SG1 computes a 2-bit Hamming distance using three bitwise steps:

```c
uint64_t diff = query ^ component;               // 1. XOR: mark differing bits
uint64_t norm = (diff | (diff >> 1)) & 0x5555555555555555ULL; // 2. Collapse each 2-bit pair to its LSB
int mismatches = __popcll(norm);                 // 3. Count differing base pairs
```

- Step 1: XOR isolates positions where query and component differ.
- Step 2: OR with a right-shift ensures that if either bit in a 2-bit base pair differs, the LSB of that pair is set. The mask `0x5555...` isolates only those LSBs.
- Step 3: Popcount gives the total number of mismatched base pairs.

## Getting Started
### Prerequisites
* NVIDIA GPU (Compute Capability 8.0+)
* CUDA Toolkit 11.8+
* Python 3.9+ / CuPy (falls back to CPU/NumPy if no GPU is available)

### Installation
```bash
git clone https://github.com/sumathigenomics/SG1_SENTINEL.git
cd SG1_SENTINEL
pip install -r requirements.txt
