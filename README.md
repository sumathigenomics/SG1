# SG1
High-speed bit-packed CUDA engine for genetic circuit orthogonality.
# SG1: High-Speed Bit-Packed CUDA Engine for Synthetic Biology

**Developed by:** Sumathi Genomics Co., Ltd. (Thailand) & Saminathan Industries Pte. Ltd. (Singapore)  
**Lead Researcher:** Dr. Saminathan Sivaprakasham Murugesan (saminathan@u.nus.edu)

## Overview
SG1 is a hardware-accelerated "Biological Design Rule Checker" (DRC). It utilizes 2-bit DNA vectorization and NVIDIA CUDA kernels to perform ultra-fast orthogonality verification and cross-talk analysis for synthetic genetic circuits.

## Key Features
* **Bit-Packed DNA Encoding:** Reduces genomic footprint by 75%, enabling entire host genomes to reside in GPU L2 cache.
* **SG1 Mismatch Transform:** A proprietary bitwise logic that normalizes 2-bit Hamming distances in a single clock cycle.
* **Orthogonality Matrix Mode:** All-vs-all comparison of gRNA regulators against synthetic circuit components.
* **Sub-second Latency:** Full human genome (GRCh38) scans in <800ms.

## Core Logic (The SG1 Transform)
To ensure 1:1 biological fidelity in a 2-bit space, SG1 uses the following bitwise operation:
`V = (X ^ (X >> 1)) & 0x5555555555555555;`
`mismatches = __popcll(V);`

## Getting Started
### Prerequisites
* NVIDIA GPU (Compute Capability 8.0+)
* CUDA Toolkit 11.8+
* Python 3.9+ / CuPy

### Installation
```bash
git clone [https://github.com/SumathiGenomics/SG1.git](https://github.com/SumathiGenomics/SG1.git)
cd SG1
pip install -r requirements.txt
