from math import log2

import pytest

from rulelist.mdl.mdl_base_codes import multinomial_with_recurrence, universal_code_integers, \
    universal_code_integers_maximum


class TestMultinomialWithRecurrence:
    def test_cardinality_one(self):
        #edge case
        input_cardinality = 1
        input_n = 2
        expected_complexity = 1.0
        actual_complexity = multinomial_with_recurrence(input_cardinality,input_n)
        assert expected_complexity == pytest.approx(actual_complexity)

    def test_cardinality_two(self):
        #edge case
        input_cardinality = 2
        input_n = 1
        expected_complexity = 2.0
        actual_complexity = multinomial_with_recurrence(input_cardinality,input_n)
        assert expected_complexity == pytest.approx(actual_complexity)

    def test_cardinality_minimum(self):
        #edge case
        input_cardinality = 2
        input_n = 2
        expected_complexity = 2.5
        actual_complexity = multinomial_with_recurrence(input_cardinality,input_n)
        assert expected_complexity == pytest.approx(actual_complexity)

    def test_cardinality_big(self):
        #normal
        input_cardinality = 10
        input_n = 10000
        expected_complexity = 3597043942882793.0
        actual_complexity = multinomial_with_recurrence(input_cardinality,input_n)
        assert expected_complexity == pytest.approx(actual_complexity)

class TestUniversalCodeIntegers:
    def test_n_zero(self):
        #edge case
        input_n = 0
        expected_codelength = 0
        codelength = universal_code_integers(input_n)
        assert expected_codelength == pytest.approx(codelength)

    def test_n_negative(self):
        # error
        input_n = -1
        with pytest.raises(ValueError) as exception_info:  # store the exception
            universal_code_integers(input_n)(input_n)
        assert exception_info.match("n should be larger than 0. The value was: -1")

    def test_n_one(self):
        #edge case
        input_n = 1
        expected_codelength = log2(2.865064)
        codelength = universal_code_integers(input_n)
        assert expected_codelength == pytest.approx(codelength)


    def test_n_large(self):
        input_n = 1000000
        expected_codelength = 29.06176716082425
        codelength = universal_code_integers(input_n)
        assert expected_codelength == pytest.approx(codelength)

class TestUniversalCodeIntegersMaximum:
    def test_n_one(self):
        #edge case
        input_n = 1
        input_maximum = 1
        expected_codelength = 0
        codelength = universal_code_integers_maximum(input_n,input_maximum)
        assert expected_codelength == pytest.approx(codelength)

    def test_n_negative(self):
        input_n1 = 1
        input_maximum = 2
        input_n2 = 2
        expected_probability_total = 1
        actual_probability_total = 2**-universal_code_integers_maximum(input_n1,input_maximum)+\
                           2**-universal_code_integers_maximum(input_n2, input_maximum)
        assert expected_probability_total == pytest.approx(actual_probability_total)
