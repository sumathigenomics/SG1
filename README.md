# SG1 SENTINEL
### High-Speed Bit-Packed CUDA Engine for Genetic Circuit Orthogonality

> Built by **Sumathi Genomics** — accelerating the future of synthetic biology.

---

## Overview

SG1 SENTINEL is a hardware-accelerated **Biological Design Rule Checker (DRC)** for synthetic genetic circuits. It uses 2-bit DNA vectorization and NVIDIA CUDA kernels to perform ultra-fast orthogonality verification and cross-talk analysis — helping synthetic biologists design safer, more reliable genetic circuits.

Whether you're designing CRISPR regulators, toggle switches, or large-scale gene circuits, SG1 SENTINEL ensures your components don't interfere with each other or the host genome.

---

## Key Features

- **Bit-Packed DNA Encoding** — Reduces genomic memory footprint by 75%, enabling entire host genomes to reside in GPU L2 cache.
- **SG1 Mismatch Transform** — Proprietary bitwise logic that normalizes 2-bit Hamming distances in a single clock cycle.
- **Orthogonality Matrix Mode** — All-vs-all comparison of gRNA regulators against synthetic circuit components.
- **Sub-second Latency** — Full human genome (GRCh38) scans in under 800ms.
- **CPU Fallback** — Runs on any machine with NumPy, no GPU required for development and testing.

---

## Core Logic (The SG1 Transform)

For each query-component pair, SG1 computes a 2-bit Hamming distance in three bitwise steps:

```c
uint64_t diff = query ^ component;                               // 1. XOR: mark differing bits
uint64_t norm = (diff | (diff >> 1)) & 0x5555555555555555ULL;   // 2. Collapse each 2-bit pair to its LSB
int mismatches = __popcll(norm);                                 // 3. Count differing base pairs
```

- Step 1: XOR isolates positions where query and component differ.
- Step 2: OR with right-shift collapses each 2-bit base pair — if either bit differs, the LSB is set. The mask `0x5555...` isolates only those LSBs.
- Step 3: Popcount gives the total number of mismatched base pairs.

---

## Getting Started

### Prerequisites
- NVIDIA GPU (Compute Capability 8.0+)
- CUDA Toolkit 11.8+
- Python 3.9+
- CuPy (optional — falls back to CPU/NumPy automatically)

### Installation

```bash
git clone https://github.com/sumathigenomics/SG1_SENTINEL.git
cd SG1_SENTINEL
pip install -r requirements.txt
```

### Run an Example

```bash
python examples/check_orthogonality.py
python examples/genomic_safety_scan.py
```

---

## About Sumathi Genomics

**Sumathi Genomics Co., Ltd.** is a synthetic biology technology company headquartered in Thailand, with operations in Singapore through **Saminathan Industries Pte. Ltd.** We build high-performance computational tools for genomics, genetic circuit design, and biological safety verification.

SG1 SENTINEL is part of our broader mission to make precision synthetic biology accessible, fast, and safe.

---

## Contact & Collaboration

We welcome research collaborations, licensing inquiries, and custom deployments.

| | |
|---|---|
| Website | [www.sumathigenomics.com](http://www.sumathigenomics.com) |
| Lab Email | [lab@sumathigenomics.com](mailto:lab@sumathigenomics.com) |
| Lead Researcher | Dr. Saminathan Sivaprakasham Murugesan |
| Academic Contact | [saminathan@u.nus.edu](mailto:saminathan@u.nus.edu) |

---

## License

See [LICENSE](LICENSE) for details.

---

*Developed by Sumathi Genomics Co., Ltd. (Thailand) & Saminathan Industries Pte. Ltd. (Singapore)*
