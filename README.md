# SG1 SENTINEL
High-speed bit-packed CUDA engine for genetic circuit orthogonality.

## Overview
SG1 SENTINEL is a hardware-accelerated biological design-rule checker. It uses 2-bit DNA vectorization and NVIDIA CUDA kernels to perform orthogonality verification and cross-talk analysis for synthetic genetic circuits.

## Key Features
* **Bit-Packed DNA Encoding:** Reduces genomic footprint by ~75% versus plain text representations.
* **SG1 Mismatch Transform:** Fast bitwise normalization of packed 2-bit mismatches.
* **Orthogonality Matrix Mode:** All-vs-all comparison of gRNA regulators against synthetic circuit components.
* **GPU-accelerated scanning:** Suitable for large circuit libraries and genome-scale screens.

## Core Logic (SG1 mismatch transform)
For each packed 64-bit sequence block:

```c
uint64_t diff = q ^ c;
uint64_t norm = (diff | (diff >> 1)) & 0x5555555555555555ULL;
int mismatches = __popcll(norm);
```

This matches the CUDA kernel implementation used by `SG1Engine`.

## Getting Started
### Prerequisites
* NVIDIA GPU with CUDA support
* CUDA Toolkit 11.8+
* Python 3.9+

### Installation
```bash
git clone https://github.com/SumathiGenomics/SG1_SENTINEL.git
cd SG1_SENTINEL
pip install -r requirements.txt
```

### Run examples
```bash
python examples/check_orthogonality.py
python examples/genomic_safety_scan.py
```

### Quick self-check (no GPU required)
```bash
python -m unittest tests/test_engine_reference.py
```

This validates the SG1 transform and thresholding logic via a CPU reference implementation in `engine.py`.

