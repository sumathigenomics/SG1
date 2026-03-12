import numpy as np
from engine import SG1Engine

# 1. Setup the SG1 Engine
sg1 = SG1Engine()

# 2. Define a "Synthetic Circuit" (Simplified 2-bit DNA)
# In a real case, these would be loaded from a .fasta file
# Let's assume these are 4 parts of a Toggle Switch circuit
circuit_parts = np.array([
    0x5555555555555555, # Part A: Promoter 1
    0xAAAAAAAAAAAAAAAA, # Part B: Repressor 1
    0xCCCCCCCCCCCCCCCC, # Part C: Promoter 2
    0x3333333333333333  # Part D: Repressor 2
], dtype=np.uint64)

# 3. Define our gRNA Regulators
queries = np.array([
    0x5555555555555550, # Guide 1 (Designed for Part A)
    0x9999999999999999  # Guide 2 (A random "Alien" guide)
], dtype=np.uint64)

print("--- SG1 Circuit Debugger ---")
print("Scanning circuit for cross-talk...")

# 4. Run the Search
results = sg1.check_orthogonality(circuit_parts, queries, max_m=5)

# 5. Display the Orthogonality Matrix
# Rows = Guides, Columns = Circuit Parts
print("\nOrthogonality Matrix (Mismatches):")
print(results)

print("\nInterpretation:")
print("- A value of 0-5 means 'Warning: Potential Cross-talk'")
print("- A value of -1 means 'Safe: Components are Orthogonal'")
