import numpy as np
from gmpy2 import mpz

from rulelist.util.bitset_operations import indexes2bitset, bitset2indexes


class TestIndexes2Bitset:
    def test_allconsecutive_array(self):
        test_input = np.array([0,1, 2, 3],dtype = np.int32)
        expected_bitarray = mpz(15)
        actual_bitarray = indexes2bitset(test_input)
        assert expected_bitarray == actual_bitarray

    def test_empty_array(self):
        test_input = np.array([],dtype = np.int32)
        expected_bitarray = mpz(0)
        actual_bitarray = indexes2bitset(test_input)
        assert expected_bitarray == actual_bitarray

    def test_oneinbeggining_array(self):
        test_input = np.array([0],dtype = np.int32)
        expected_bitarray = mpz(1)
        actual_bitarray = indexes2bitset(test_input)
        assert expected_bitarray == actual_bitarray

    def test_oneatend_array(self):
        test_input = np.array([4],dtype = np.int32)
        expected_bitarray = mpz(16)
        actual_bitarray = indexes2bitset(test_input)
        assert expected_bitarray == actual_bitarray

    def test_dtypefloat_array(self):
        test_input = np.array([4],dtype = np.float64)
        expected_bitarray = mpz(16)
        actual_bitarray = indexes2bitset(test_input)
        assert expected_bitarray == actual_bitarray


class TestBitset2Indexes:
    def test_allconsecutive_array(self):
        test_input = mpz(15)
        expected_bitarray = np.array([0,1, 2, 3],dtype = np.int32)
        actual_bitarray = bitset2indexes(test_input)
        np.testing.assert_array_equal(expected_bitarray,actual_bitarray)

    def test_empty_array(self):
        test_input = mpz(0)
        expected_bitarray = np.array([], dtype = np.int32)
        actual_bitarray = bitset2indexes(test_input)
        np.testing.assert_array_equal(expected_bitarray,actual_bitarray)

    def test_oneinbeggining_array(self):
        test_input = mpz(1)
        expected_bitarray = np.array([0])
        actual_bitarray = bitset2indexes(test_input)
        np.testing.assert_array_equal(expected_bitarray,actual_bitarray)

    def test_oneatend_array(self):
        test_input = mpz(16)
        expected_bitarray = np.array([4], dtype = np.int32)
        actual_bitarray = bitset2indexes(test_input)
        np.testing.assert_array_equal(expected_bitarray,actual_bitarray)
