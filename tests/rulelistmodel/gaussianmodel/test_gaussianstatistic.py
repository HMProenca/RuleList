import numpy as np
import pandas as pd
import pytest
from gmpy2 import bit_mask

from rulelist.datastructure.data import Data
from rulelist.rulelistmodel.gaussianmodel.gaussianstatistic import find2points, GaussianFixedStatistic, \
    GaussianFreeStatistic


class TestFind2Points(object):
    def test_2points_and_mean_start(self):
        input_values = np.arange(100)
        input_mean = 0
        input_bigvalue = 200

        expected_2points = np.array([0,1])
        expected_difference = [0,1]
        actual_2poits, acttual_difference= find2points(input_values, input_mean,input_bigvalue)

        np.testing.assert_array_equal(expected_2points,actual_2poits)
        assert expected_difference == acttual_difference

    def test_2points_and_mean_end(self):
        input_values = np.arange(100)
        input_mean = 99
        input_bigvalue = 200

        expected_2points = [99,98]
        expected_difference = [0,1]

        actual_2poits, acttual_difference= find2points(input_values, input_mean,input_bigvalue)

        np.testing.assert_array_equal(expected_2points,actual_2poits)
        assert expected_difference == acttual_difference

    def test_2points_and_mean_only2values(self):
        input_values = np.array([0 if i < 50 else 1 for i in range(100)])
        input_mean = 99
        input_bigvalue = 200

        expected_2points = [1, 0]
        expected_difference = [98, 99]
        actual_2poits, acttual_difference = find2points(input_values, input_mean,input_bigvalue)

        np.testing.assert_array_equal(expected_2points,actual_2poits)
        assert expected_difference == acttual_difference

    def test_2points_and_mean_only1value(self):
        input_values = np.ones(100)
        input_mean = 99
        input_bigvalue = 200

        expected_2points = [1, 200]
        expected_difference = [98, 101]
        actual_2poits, acttual_difference = find2points(input_values, input_mean,input_bigvalue)

        np.testing.assert_array_equal(expected_2points,actual_2poits)
        assert expected_difference == acttual_difference

@pytest.fixture
def constant_parameters():
    input_n_cutpoints = 5
    input_discretization = "static"
    input_target_data = "gaussian"
    input_minsupp = 0
    dictinput = {"attribute1": np.arange(100),
                 "attribute2": np.array(["below50" if i < 50 else "above49" for i in range(100)])}
    input_input_data = pd.DataFrame(data=dictinput)
    yield input_input_data, input_n_cutpoints, input_discretization, input_target_data, input_minsupp


@pytest.fixture
def generate_inputvalues_one_target(constant_parameters):
    input_input_data, input_n_cutpoints, input_discretization, input_target_data, input_minsupp = constant_parameters
    # targets
    np.random.seed(seed=42)
    dictoutput = {"target1": np.arange(100)}
    input_output_data = pd.DataFrame(data=dictoutput)
    data_class = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    input_bitarray_for_statistic = bit_mask(data_class.number_instances)
    yield data_class, input_bitarray_for_statistic

@pytest.fixture
def generate_inputvalues_multiple_targets(constant_parameters):
    input_input_data, input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters
    # targets
    np.random.seed(seed=42)
    dictoutput = {"target1": np.arange(100),
                  "target2": np.ones(100),
                  "target3": np.arange(100)[100::-1],
                  "target4": np.random.rand(100)*100}
    input_output_data = pd.DataFrame(data=dictoutput)
    data_class = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    input_bitarray_for_statistic = bit_mask(data_class.number_instances)
    yield data_class, input_bitarray_for_statistic

class TestGaussianFixedStatistic(object):
    def test_4targets(self,generate_inputvalues_multiple_targets):
        data_class, input_bitarray_for_statistic = generate_inputvalues_multiple_targets

        statistic = GaussianFixedStatistic(data_class)
        statistic.replace_stats(data_class,input_bitarray_for_statistic)

        expected_usage = 100
        expected_mean = [49.5, 1, 49.5, np.mean(data_class.target_data.values[:,3]) ]
        expected_variance = [833.25, 0, 833.25, np.var(data_class.target_data.values[:,3]) ]
        expected_sizelist = 4

        assert expected_usage == statistic.usage
        assert expected_mean ==  pytest.approx(statistic.mean)
        assert expected_variance == pytest.approx(statistic.variance)
        assert expected_sizelist == len(statistic.mean)
        assert expected_sizelist == len(statistic.variance)
        assert expected_sizelist == len(statistic.rss)


    def test_onetarget(self,generate_inputvalues_one_target):
        data_class, input_bitarray_for_statistic = generate_inputvalues_one_target

        statistic = GaussianFixedStatistic(data_class)
        statistic.replace_stats(data_class,input_bitarray_for_statistic)

        expected_usage = 100
        expected_mean = [49.5]
        expected_variance = [833.25]
        expected_sizelist = 1

        assert expected_mean ==  pytest.approx(statistic.mean)
        assert expected_variance == pytest.approx(statistic.variance)
        assert expected_variance == statistic.variance
        assert expected_sizelist == len(statistic.mean)
        assert expected_sizelist == len(statistic.variance)
        assert expected_sizelist == len(statistic.rss)


class TestGaussianFreeStatistic(object):
    def test_4targets(self,generate_inputvalues_multiple_targets):
        data_class, input_bitarray_for_statistic = generate_inputvalues_multiple_targets

        statistic = GaussianFreeStatistic(data_class)
        statistic.replace_stats(data_class,input_bitarray_for_statistic)

        expected_usage = 100
        expected_mean = [49.5, 1, 49.5, np.mean(data_class.target_data.values[:,3]) ]
        expected_variance = [833.25, 0, 833.25, np.var(data_class.target_data.values[:,3]) ]
        expected_sizelist = 4


        assert expected_usage == statistic.usage
        assert expected_mean ==  pytest.approx(statistic.mean,abs=3e1)
        assert expected_variance == pytest.approx(statistic.variance,abs=5e1)
        assert expected_sizelist == len(statistic.mean)
        assert expected_sizelist == len(statistic.variance)
        assert expected_sizelist == len(statistic.rss)
        assert expected_sizelist == len(statistic.mean_2points)
        assert expected_sizelist == len(statistic.variance_2points)
        assert expected_sizelist == len(statistic.rss_2points)
        assert expected_sizelist == len(statistic.mean_dataset)
        assert expected_sizelist == len(statistic.variance_dataset)
        assert expected_sizelist == len(statistic.rss_2dataset)

    #def test_onetarget(self,generate_inputvalues_one_target):


#class TestComputeValuesData: use mock!

#class TestComputeStatisticsAllrules:

