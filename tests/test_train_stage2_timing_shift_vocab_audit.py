import unittest

from train.stage_2.events.audit_timing_shift_vocab import (
    TimingSection,
    best_grid_match,
    decompose_interval_to_grid_tokens,
)


def _token_pairs(result):
    return [(token.divisor, token.k) for token in result.tokens]


class Stage2TimingShiftVocabAuditTests(unittest.TestCase):
    def test_constant_bpm_exact_binary_subdivisions(self) -> None:
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=500.0)]

        self.assertTrue(best_grid_match(500.0, sections, [1, 2, 4, 8, 16]).exact)
        self.assertTrue(best_grid_match(250.0, sections, [1, 2, 4, 8, 16]).exact)
        self.assertTrue(best_grid_match(125.0, sections, [1, 2, 4, 8, 16]).exact)

        result = decompose_interval_to_grid_tokens(0.0, 125.0, sections, [1, 2, 4, 8, 16], k_max=16)
        self.assertTrue(result.ok)
        self.assertEqual(_token_pairs(result), [(4, 1)])

    def test_triplet_grid_exactness(self) -> None:
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=600.0)]

        self.assertFalse(best_grid_match(200.0, sections, [1, 2, 4, 8, 16]).exact)
        self.assertTrue(best_grid_match(200.0, sections, [1, 2, 3, 4, 8, 16]).exact)

        result = decompose_interval_to_grid_tokens(0.0, 200.0, sections, [1, 2, 3, 4], k_max=16)
        self.assertTrue(result.ok)
        self.assertEqual(_token_pairs(result), [(3, 1)])

    def test_bpm_change_boundary_splits_decomposition(self) -> None:
        sections = [
            TimingSection(offset_ms=0.0, beat_length_ms=500.0),
            TimingSection(offset_ms=1000.0, beat_length_ms=250.0),
        ]

        result = decompose_interval_to_grid_tokens(500.0, 1250.0, sections, [1, 2, 4], k_max=16)
        self.assertTrue(result.ok)
        self.assertEqual(_token_pairs(result), [(1, 1), (1, 1)])
        self.assertEqual([token.section_index for token in result.tokens], [0, 1])

    def test_long_shift_uses_greedy_canonical_decomposition(self) -> None:
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=500.0)]

        result = decompose_interval_to_grid_tokens(0.0, 5000.0, sections, [1, 2, 4], k_max=4)
        self.assertTrue(result.ok)
        self.assertEqual(_token_pairs(result), [(1, 4), (1, 4), (1, 2)])

    def test_ambiguous_quarter_vs_two_eighths_prefers_coarser_divisor(self) -> None:
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=400.0)]

        result = decompose_interval_to_grid_tokens(0.0, 100.0, sections, [4, 8], k_max=16)
        self.assertTrue(result.ok)
        self.assertEqual(_token_pairs(result), [(4, 1)])


if __name__ == "__main__":
    unittest.main()
