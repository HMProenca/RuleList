from gmpy2 import mpz, bit_set, xmpz
import numpy as np

def indexes2bitset(vector2transform: np.ndarray) -> mpz:
    """ Transforms a numpy vector of indexes into a bitset (gmpy2 multiple precision integer).

    """
    bit_array = mpz()
    for index in vector2transform:
        bit_array = bit_array.bit_set(int(index))
    return bit_array

def compute_index(bitset2transform: mpz) -> np.ndarray:
    """ Transforms a bitset (gmpy2 multiple precision integer) into a numpy array of indexes.

    """
    indexes = np.array([ix for ix, x in enumerate(reversed(bin(bitset2transform)[2:])) if x == '1'],
                       dtype = np.int32)
    return indexes

def bitset2indexes(bitarray):
    bitarray_iterable = xmpz(bitarray)
    idx_subgroup = [*bitarray_iterable.iter_set()]
    return idx_subgroup