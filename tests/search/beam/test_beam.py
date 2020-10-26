import numpy as np
import pytest

from rulelist.search.beam.beam import Beam


@pytest.fixture
def start_beam():
    input_width = 4
    beam = Beam(input_width)
    expected_patterns = [() for w in range(input_width)]
    expected_array_score = np.full(input_width, np.NINF)
    expected_min_score = np.NINF
    expected_min_index = 0

    assert  input_width == beam.beam_width
    assert set(expected_patterns) == set(beam.patterns)
    np.testing.assert_equal(expected_array_score,beam.array_score)
    assert expected_min_score == beam.min_score
    assert expected_min_index == beam.min_index

class TestBeam:
    def test_init(self):
        input_width = 4
        beam = Beam(input_width)
        expected_patterns = [[] for w in range(input_width)]
        expected_array_score = np.full(input_width, np.NINF)
        expected_set_patterns = [set() for w in range(input_width)]
        expected_min_score = np.NINF
        expected_min_index = 0

        assert  input_width == beam.beam_width
        assert sorted(expected_patterns) == sorted(beam.patterns)
        np.testing.assert_equal(expected_array_score,beam.array_score)
        assert expected_set_patterns == beam.set_patterns
        assert expected_min_score == beam.min_score
        assert expected_min_index == beam.min_index

    @pytest.mark.skip(reason="Needs to add the beam.set_patterns")
    def test_replace(self):
        input_width = 4
        beam = Beam(input_width)
        input_pattern = ["test"]
        input_score = 10.0

        beam.replace(input_pattern,input_score)

        expected_patterns = [["test"]] + [() for w in range(input_width-1)]
        expected_array_score = np.full(input_width, np.NINF)
        #expected_set_patterns = [set() for w in range(input_width)]
        expected_array_score[0] = input_score
        expected_min_score = np.NINF
        expected_min_index = 1

        assert input_width == beam.beam_width
        assert expected_patterns == beam.patterns
        np.testing.assert_equal(expected_array_score, beam.array_score)
        assert expected_min_score == beam.min_score
        assert expected_min_index == beam.min_index

    @pytest.mark.skip(reason="Needs to add the beam.set_patterns")
    def test_replace_and_clean(self):
        input_width = 4
        beam = Beam(input_width)
        input_pattern = ["test"]
        input_score = 10.0
        beam.replace(input_pattern, input_score)
        beam.clean()

        expected_patterns = [[] for w in range(input_width)]
        expected_array_score = np.full(input_width, np.NINF)
        expected_set_patterns = [set() for w in range(input_width)]
        expected_min_score = np.NINF
        expected_min_index = 0

        assert input_width == beam.beam_width
        assert expected_patterns == beam.patterns
        np.testing.assert_equal(expected_array_score, beam.array_score)
        assert expected_set_patterns == beam.set_patterns
        assert expected_min_score == beam.min_score
        assert expected_min_index == beam.min_index