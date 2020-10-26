# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:34:06 2019

@author: gathu
"""

from functools import lru_cache
from math import log, ceil, sqrt, log2

from scipy.special import comb, gammaln

from mdlrulelist.util.build.extra_maths import log2_0


@lru_cache(maxsize=20000,typed=True)
def log_multinomial(cardinality, n):
    return log2_0(multinomial_with_recurrence(cardinality, n))

def multinomial_with_recurrence(cardinality, n):
    """ Computes the Normalized Maximum Likelihood (NML) code length
    cardinality - number of categories for a single-nominal or multinomial distribution
    n - number of points / samples
    complexity  - COMP(cardinality,n) - the complexity (without logarithm)
    """
    complexity = 1.0
    b = 1.0
    d = 10   # seven digit precision
    if cardinality == 1:
        complexity = 1.0
    elif n == 0: 
        complexity = 0
    else:
        bound = int(ceil(2 + sqrt(2 * n * d * log(10))))  # using equation (38)
        for k in range(1, bound + 1):
            b = (n - k + 1) / n * b
            complexity += b
        old_sum = 1.0
        for j in range(3, cardinality + 1):
            new_sum = complexity + (n * old_sum) / (j - 2)
            old_sum = complexity
            complexity = new_sum
    return complexity

@lru_cache(maxsize=20000,typed=True)
def universal_code_integers(value: int) -> float:
    """ computes the universal code of integers 
    """
    const =  2.865064
    logsum = log2(const)
    cond = True # condition
    if value == 0:
        logsum = 0
    elif value > 0:
        while cond: # Recursive log
            value = log2(value)
            cond = value > 0.000001
            if value < 0.000001:
                break
            logsum += value
    elif value < 0:
        raise ValueError('n should be larger than 0. The value was: {}'.format(value))
    return logsum

@lru_cache(maxsize=20000,typed=True)
def log2_gamma_half(n: int):
    le2 =  0.6931471805599453 # log(2)
    return gammaln(n / 2) / le2 if n > 0 else 0

def universal_code_integers_maximum(n: int, maximum : int) -> float:
    """ computes the universal code of integers when there is a known maximum integer
    This is equivalent to applying the maximum entropy principle knowing the maximum,
    and it equalitarian  division of the non-used probability (the ones after the maximum)
    by all the used number (1 until maximum).
    """
    probability_until_max = sum([2**-universal_code_integers(n_aux)  for n_aux in range(1,maximum+1)])
    probability_left = 1 - probability_until_max
    probability_n = 2**-universal_code_integers(n)+ probability_left/maximum
    logsum = -log2(probability_n)
    return logsum

def uniform_code(n: int) -> float:
    return log2(n) if n != 0 else 0

def uniform_combination_code(n: int, maximum: int) -> float:
    return log2(comb(maximum, n))