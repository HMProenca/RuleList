# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:09:04 2020

@author: gathu
"""
import math
from math import pi, log2

from numpy import inf

from rulelist.mdl.mdl_base_codes import log2_gamma_half
from rulelist.rulelistmodel.gaussianmodel.gaussianstatistic import GaussianFixedStatistic


def gaussian_bayesian_encoding(n: int,variance : float,log_gamma_n: float):
    """ Computes the Bayesian encoding of single-numeric target with mean and variance unknown.


    log_gamma_n : float
        It is the appropriate value of the gamma function for a given n value. In the case of the Bayesian encoding
        of the paper it is log2( Gamma(n/2) ).

    """
    if n < 2 or variance == 0:
        length = inf
    else:
        length = 1 + n/2*log2(pi) - log_gamma_n + 0.5*log2(n+1) + n/2*log2(n*variance)
    return length


def gaussian_fixed_encoding(n: int, rss: float, variance: float):
    """ Computes the encoding of a single-numeric target when the mean and variance are fixed to a value.

    rss : float
        Residual Sum of Squares with a fixed mean.
    variance: float
        Fixed variance of the Gaussian distribution.
    """
    if variance == 0:
        length = inf
    else:
        log2_e = 1.4426950408889634
        length = 0.5*n*log2(2 * pi * variance)
        length += 0.5 * log2_e * rss / variance
    return length

def length_rule_free_gaussian(rulelist : classmethod, statistics : classmethod):
    """ Computes alpha_gain of adding one rule that does not have fixed statistics.

    """
    if any(statistics.variance) == 0 or statistics.usage <= 2:
        codelength = inf
    else:
        loggamma_usg = log2_gamma_half(statistics.usage)
        loggamma_2 = log2_gamma_half(2)
        number_of_targets = len(statistics.mean)
        l_bayesian_all = sum([gaussian_bayesian_encoding(statistics.usage, statistics.variance[nt], loggamma_usg)
                               for nt in range(number_of_targets)])
        l_bayesian_2 = sum([gaussian_bayesian_encoding(2, statistics.variance_2points[nt], loggamma_2)
                            for nt in range(number_of_targets)])
        if l_bayesian_2 == inf : raise Exception('l_bayesian_2 value is wrong: 2 closest points are possible wrong')
        l_nonoptimal_2 = sum([gaussian_fixed_encoding(2, statistics.rss_2dataset[nt],
                                                      statistics.variance_dataset[nt])
                              for nt in range(number_of_targets)])
        if l_nonoptimal_2 == inf : raise Exception('l_nonoptimal_2 value is wrong')
        codelength = l_bayesian_all - l_bayesian_2 + l_nonoptimal_2
    return codelength


def length_rule_fixed_gaussian(rulelist : classmethod, statistics : GaussianFixedStatistic):
    """ Computes alpha_gain of one rule that does not have fixed statistics.

    """
    number_of_targets = len(statistics.mean)
    l_fixed = sum([gaussian_fixed_encoding(statistics.usage, statistics.rss[nt], statistics.variance[nt])
                   for nt in range(number_of_targets)])
    return l_fixed


