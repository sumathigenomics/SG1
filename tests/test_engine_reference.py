import unittest

import numpy as np

from engine import check_orthogonality_cpu, sg1_mismatch_distance


class TestSG1Reference(unittest.TestCase):
    def test_zero_distance_for_identical_blocks(self):
        block = np.uint64(0xAAAAAAAAAAAAAAAA)
        self.assertEqual(sg1_mismatch_distance(block, block), 0)

    def test_distance_counts_two_bit_nucleotide_changes(self):
        a = np.uint64(0x0)
        c = np.uint64(0x5555555555555555)
        self.assertEqual(sg1_mismatch_distance(a, c), 32)

    def test_cpu_orthogonality_thresholding(self):
        circuit = np.array([
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
        ], dtype=np.uint64)
        queries = np.array([
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
        ], dtype=np.uint64)

        result = check_orthogonality_cpu(circuit, queries, max_m=0)
        expected = np.array([
            [0, -1],
            [-1, 0],
        ], dtype=np.int32)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
