from math import log, pi, exp, log2

import numpy as np
import pytest
from scipy.special.cython_special import gammaln

from rulelist.rulelistmodel.gaussianmodel.mdl_gaussian import gaussian_bayesian_encoding, gaussian_fixed_encoding, \
    length_rule_free_gaussian, length_rule_fixed_gaussian


class TestGaussianBayesianEncoding:
    def test_n_zero(self):
        input_n = 0
        input_variance = 100
        input_log_gamma_n =  gammaln(input_n / 2) / log(2)
        actual_length = gaussian_bayesian_encoding(input_n, input_variance, input_log_gamma_n)

        expected_length = np.inf
        assert expected_length == actual_length

    def test_n_one(self):
        input_n = 1
        input_variance = 100
        input_log_gamma_n = gammaln(input_n / 2) / log(2)
        actual_length = gaussian_bayesian_encoding(input_n, input_variance, input_log_gamma_n)

        expected_length = np.inf
        assert expected_length == actual_length

    def test_n_two(self):
        input_n = 2
        input_variance = 100
        input_log_gamma_n = gammaln(input_n / 2) / log(2)
        actual_length = gaussian_bayesian_encoding(input_n, input_variance, input_log_gamma_n)

        expected_length = 11.087833569607621
        assert expected_length == pytest.approx(actual_length)

    def test_variance_zero(self):
        input_n = 2
        input_variance = 0
        input_log_gamma_n = gammaln(input_n / 2) / log(2)
        actual_length = gaussian_bayesian_encoding(input_n, input_variance, input_log_gamma_n)

        expected_length = np.inf
        assert expected_length == actual_length

    def test_all_large(self):
        input_n = 100000
        input_variance = 100000
        input_log_gamma_n = gammaln(input_n / 2) / log(2)
        actual_length = gaussian_bayesian_encoding(input_n, input_variance, input_log_gamma_n)

        expected_length = 1035207.3661371232
        assert expected_length == pytest.approx(actual_length)


class TestGaussianFixedEncoding:
    def test_variance_zero(self):
        input_n = 10
        input_variance = 0
        input_rss = 100
        actual_length = gaussian_fixed_encoding(input_n, input_rss,input_variance)

        expected_length = np.inf
        assert expected_length == actual_length

    def test_exponential_result(self):
        input_n = 2
        input_variance = 1/(2*pi)
        input_rss = input_n*input_variance
        actual_length = gaussian_fixed_encoding(input_n, input_rss, input_variance)

        expected_length = log2(exp(1))
        assert expected_length == pytest.approx(actual_length)

    def test_exponential_large(self):
        input_n = 100000
        input_variance = 1
        input_rss= 1*input_n
        expected_length = 0.5*input_n*(log2(2*pi*input_variance*exp(1)))
        actual_length = gaussian_fixed_encoding(input_n, input_rss, input_variance)
        assert expected_length == pytest.approx(actual_length)

@pytest.fixture
def makemockrulelist():
    class MockRulelist:
        def __init__(self):
            pass
    yield MockRulelist()

@pytest.fixture
def makemockgaussianfixed_onetarget():
    class MockGaussianFixedStatistic:
        def __init__(self):
            self.usage = 100000
            self.mean = [0]
            self.variance = [2]
            self.rss = [self.usage*self.variance[0]]
    yield MockGaussianFixedStatistic()

@pytest.fixture
def makemockgaussianfixed_twotarget():
    class MockGaussianFixedStatistic:
        def __init__(self):
            self.usage = 100000
            self.mean = [0,10]
            self.variance = [2,4]
            self.rss = [self.usage*self.variance[0],self.usage*self.variance[1]]
    yield MockGaussianFixedStatistic()


class TestLengthRuleFixedGaussian:
    def test_onetarget(self,makemockrulelist,makemockgaussianfixed_onetarget):
        mockgaussianfixedstatistic = makemockgaussianfixed_onetarget
        mockrulelist = makemockrulelist
        expected_length = 0.5*100000*(log2(2*pi*2*exp(1)))
        actual_length = length_rule_fixed_gaussian(mockrulelist,mockgaussianfixedstatistic)

        assert expected_length == pytest.approx(actual_length)

    def test_twotarget(self, makemockrulelist, makemockgaussianfixed_twotarget):
        mockgaussianfixedstatistic = makemockgaussianfixed_twotarget
        mockrulelist = makemockrulelist
        expected_length = 0.5 * 100000 * (log2(2 * pi * 2 * exp(1))) + \
                          0.5 * 100000 * (log2(2 * pi * 4 * exp(1)))
        actual_length = length_rule_fixed_gaussian(mockrulelist, mockgaussianfixedstatistic)
        assert expected_length == pytest.approx(actual_length)




@pytest.fixture
def makemockgaussianfree_onetarget():
    class MockGaussianFreeStatistic:
        def __init__(self):
            self.usage = 100000
            self.mean = [0]
            self.variance = [3]
            self.rss = [self.usage*self.variance]
            self.mean_2points = [0]
            self.variance_2points = [1]
            self.rss_2points = [2]
            self.mean_dataset = [0]
            self.variance_dataset = [1]
            self.rss_2dataset = [2]

    yield MockGaussianFreeStatistic()

@pytest.fixture
def makemockgaussianfree_multipletarget_onlynonoptimalcode_2points():
    class MockGaussianFreeStatistic:
        def __init__(self):
            self.usage = 2
            self.mean = [0, 0]
            self.variance = [1, 1]
            self.rss = [2, 2]
            self.mean_2points = [0, 0]
            self.variance_2points = [1, 1]
            self.rss_2points = [2, 2]
            self.mean_dataset = [0, 0]
            self.variance_dataset = [1, 1]
            self.rss_2dataset = [2, 2]

    yield MockGaussianFreeStatistic()

@pytest.fixture
def makemockgaussianfree_multipletarget_generic():
    class MockGaussianFreeStatistic:
        def __init__(self):
            self.usage = 100000
            self.mean = [0, 0]
            self.variance = [100, 150]
            self.rss = [self.usage* 100, self.usage * 150]
            self.mean_2points = [0, 0]
            self.variance_2points = [1, 1]
            self.rss_2points = [2, 2]
            self.mean_dataset = [0, 0]
            self.variance_dataset = [1, 1]
            self.rss_2dataset = [2, 2]

    yield MockGaussianFreeStatistic()

@pytest.fixture
def makemockgaussianfree_multipletarget_infinity():
    class MockGaussianFreeStatistic:
        def __init__(self):
            self.usage = 50
            self.mean = [0, 0]
            self.variance = [100, 0]
            self.rss = [50 * 100, 0]
            self.mean_2points = [0, 0]
            self.variance_2points = [1, 1]
            self.rss_2points = [2, 2]
            self.mean_dataset = [0, 0]
            self.variance_dataset = [1, 1]
            self.rss_2dataset = [2, 2]

    yield MockGaussianFreeStatistic()


class TestLengthFreeGaussian:
    def test_multipletargets_onetarget_simple(self, makemockrulelist,makemockgaussianfree_onetarget):
        mockgaussianfreestatistic = makemockgaussianfree_onetarget
        mockrulelist = makemockrulelist

        actual_length = length_rule_free_gaussian(mockrulelist, mockgaussianfreestatistic)
        n = 100000
        var= 3
        expected =  1 + n/2*log2(pi) - gammaln(n / 2) / log(2) + 0.5*log2(n+1) + n/2*log2(n*var)
        assert expected == pytest.approx(actual_length,rel=1e-3)

    def test_multipletargets_only2points(self, makemockrulelist,
                                                        makemockgaussianfree_multipletarget_onlynonoptimalcode_2points):
        mockgaussianfreestatistic = makemockgaussianfree_multipletarget_onlynonoptimalcode_2points
        mockrulelist = makemockrulelist

        actual_length = length_rule_free_gaussian(mockrulelist, mockgaussianfreestatistic)
        expected = np.inf
        assert expected == pytest.approx(actual_length)

    def test_multipletargets_generic(self, makemockrulelist,makemockgaussianfree_multipletarget_generic):
        mockgaussianfreestatistic = makemockgaussianfree_multipletarget_generic
        mockrulelist = makemockrulelist

        actual_length = length_rule_free_gaussian(mockrulelist, mockgaussianfreestatistic)
        n = 100000
        var1= 100
        var2= 150

        expected =  1 + n/2*log2(pi) - gammaln(n/2)/log(2) + 0.5*log2(n+1) + n/2*log2(n*var1)+\
                    1 + n / 2 * log2(pi) - gammaln(n / 2) / log(2) + 0.5 * log2(n + 1) + n / 2 * log2(n * var2)


        assert expected == pytest.approx(actual_length,rel=1e-3)

    def test_infitnity(self, makemockrulelist,makemockgaussianfree_multipletarget_infinity):
        mockgaussianfreestatistic = makemockgaussianfree_multipletarget_infinity
        mockrulelist = makemockrulelist

        actual_length = length_rule_free_gaussian(mockrulelist, mockgaussianfreestatistic)
        expected = np.inf
        assert expected == pytest.approx(actual_length)










