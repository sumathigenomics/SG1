import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from engine import SG1Engine

# Initialize SG1
sg1 = SG1Engine()

# Simulate a host genome (e.g., E. coli) bit-packed into 10,000 blocks
host_genome = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=10000, dtype=np.uint64)

# Your synthetic circuit's primary regulator
my_circuit_grna = 0x1234567890ABCDEF

print(f"Scanning Host Genome for interference from SG1...")

# Search for any site in the genome with 3 or fewer mismatches
results = sg1.check_orthogonality(host_genome, [my_circuit_grna], max_m=3)

hits = np.where(results[0] != -1)[0]

if len(hits) == 0:
    print("SUCCESS: No significant interference found in host genome.")
else:
    print(f"ALERT: Found {len(hits)} potential interference sites!")
