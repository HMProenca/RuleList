import numpy as np
import pandas as pd
import pytest
from gmpy2 import bit_mask

from rulelist.datastructure.data import Data
from rulelist.rulelistmodel.categoricalmodel.categoricalstatistic import CategoricalFixedStatistic, \
    CategoricalFreeStatistic


@pytest.fixture
def constant_parameters():
    input_n_cutpoints = 5
    input_discretization = "static"
    input_target_data = "categorical"
    input_minsupp = 0
    dictinput = {"attribute1": np.arange(100),
                 "attribute2": np.array(["below50" if i < 50 else "above49" for i in range(100)])}
    input_input_data = pd.DataFrame(data=dictinput)
    yield input_input_data, input_n_cutpoints, input_discretization, input_target_data,input_minsupp

@pytest.fixture
def generate_inputvalues_one_target(constant_parameters):
    input_input_data, input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters
    # targets
    dictoutput = {"target1": np.array(["below50" if i < 50 else "above49" for i in range(100)])}
    input_output_data = pd.DataFrame(data=dictoutput)
    data_class = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    input_bitarray_for_statistic = bit_mask(data_class.number_instances)
    yield data_class, input_bitarray_for_statistic

@pytest.fixture
def generate_inputvalues_two_targets(constant_parameters):
    input_input_data, input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters
    # targets
    dictoutput = {"target1": np.array(["below50" if i < 50 else "above49" for i in range(100)]),
                  "target2": np.array(["below25" if i < 25 else "above25" for i in range(100)])}

    input_output_data = pd.DataFrame(data=dictoutput)
    data_class = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    input_bitarray_for_statistic = bit_mask(data_class.number_instances)
    yield data_class, input_bitarray_for_statistic

class TestCategoricalFixedStatistic:
    def test_2targets(self,generate_inputvalues_two_targets):
        data_class, input_bitarray_for_statistic = generate_inputvalues_two_targets

        statistic = CategoricalFixedStatistic(data_class)
        statistic.replace_stats(data_class,input_bitarray_for_statistic)

        expected_usage = 100
        expected_number_targets = 2
        expected_usage_per_class ={"target1": {"below50":50, "above49":50 },
                                  "target2": {'below25': 25, 'above25': 75}}
        expected_number_classes = {'target1': 2, 'target2': 2}
        expected_prob_per_classes = {'target1': {'below50': 0.5, 'above49': 0.5},
                            'target2': {'below25': 0.25, 'above25': 0.75}}

        assert expected_usage == statistic.usage
        assert expected_number_targets ==  statistic.number_targets
        assert expected_usage_per_class == statistic.usage_per_class
        assert expected_number_classes == statistic.number_classes
        assert expected_prob_per_classes == statistic.prob_per_classes

    def test_1targets(self,generate_inputvalues_one_target):
        data_class, input_bitarray_for_statistic = generate_inputvalues_one_target

        statistic = CategoricalFixedStatistic(data_class)
        statistic.replace_stats(data_class,input_bitarray_for_statistic)

        expected_usage = 100
        expected_number_targets = 1
        expected_usage_per_class ={"target1": {"below50":50, "above49":50 }}
        expected_number_classes = {'target1': 2}
        expected_prob_per_classes = {'target1': {'below50': 0.5, 'above49': 0.5}}

        assert expected_usage == statistic.usage
        assert expected_number_targets ==  statistic.number_targets
        assert expected_usage_per_class == statistic.usage_per_class
        assert expected_number_classes == statistic.number_classes
        assert expected_prob_per_classes == statistic.prob_per_classes

class TestCategoricalFreeStatistic:
    def test_2targets(self,generate_inputvalues_two_targets):
        data_class, input_bitarray_for_statistic = generate_inputvalues_two_targets

        statistic = CategoricalFreeStatistic(data_class)
        statistic.replace_stats(data_class,input_bitarray_for_statistic)

        expected_usage = 100
        expected_number_targets = 2
        expected_usage_per_class ={"target1": {"below50":50, "above49":50 },
                                  "target2": {'below25': 25, 'above25': 75}}
        expected_number_classes = {'target1': 2, 'target2': 2}


        assert expected_usage == statistic.usage
        assert expected_number_targets ==  statistic.number_targets
        assert expected_usage_per_class == statistic.usage_per_class
        assert expected_number_classes == statistic.number_classes

    def test_1targets(self,generate_inputvalues_one_target):
        data_class, input_bitarray_for_statistic = generate_inputvalues_one_target

        statistic = CategoricalFreeStatistic(data_class)
        statistic.replace_stats(data_class,input_bitarray_for_statistic)

        expected_usage = 100
        expected_number_targets = 1
        expected_usage_per_class ={"target1": {"below50":50, "above49":50 }}
        expected_number_classes = {'target1': 2}

        assert expected_usage == statistic.usage
        assert expected_number_targets ==  statistic.number_targets
        assert expected_usage_per_class == statistic.usage_per_class
        assert expected_number_classes == statistic.number_classes
