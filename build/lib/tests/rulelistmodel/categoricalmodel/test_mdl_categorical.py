import numpy as np
import pandas as pd
import pytest
from gmpy2 import bit_mask

from rulelist.datastructure.data import Data
from rulelist.rulelistmodel.categoricalmodel.mdl_categorical import categorical_free_encoding, \
    categorical_fixed_encoding, \
    length_rule_free_categorical, length_rule_fixed_categorical
from rulelist.util.extra_maths import log2_0


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
    yield data_class

@pytest.fixture
def generate_inputvalues_two_targets(constant_parameters):
    input_input_data, input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters
    # targets
    dictoutput = {"target1": np.array(["below50" if i < 50 else "above49" for i in range(100)]),
                  "target2": np.array(["below99" if i < 99 else "above99" for i in range(100)])}

    input_output_data = pd.DataFrame(data=dictoutput)
    data_class = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    yield data_class

@pytest.fixture
def makemockrulelist():
    class MockRulelist:
        def __init__(self,data_class):
            self.log_prior_class = {varname:{category: -log2_0(count / data_class.number_instances)
                                             for category, count in counts.items()} for varname, counts in
                                             data_class.targets_info.counts.items()}
    yield MockRulelist

@pytest.fixture
def makemockcategoricalfixed_onetarget():
    class MockCategoricalFixedStatistic:
        def __init__(self):
            self.usage = 100
            self.number_targets = 1
            self.usage_per_class = {"target1": {"below50": 50, "above49": 50}}
            self.number_classes = {'target1': 2, 'target2': 2}
            self.prob_per_classes = {'target1': {'below50': 0.5, 'above49': 0.5}}
    yield MockCategoricalFixedStatistic()

@pytest.fixture
def makemockcategoricalfixed_twotargets():
    class MockCategoricalFixedStatistic:
        def __init__(self):
            self.usage = 100
            self.number_targets = 2
            self.usage_per_class = {"target1": {"below50": 50, "above49": 50},
                                    "target2": {'below99': 99, 'above99': 1}}
            self.number_classes = {'target1': 2, 'target2': 2}
            self.prob_per_classes = {'target1': {'below50': 0.5, 'above49': 0.5},
                                     'target2': {'below99': 0.99, 'above99': 0.01}}
    yield MockCategoricalFixedStatistic()


class TestCategoricalFreeEncoding:
    def test_2targets(self,makemockcategoricalfixed_twotargets):
        input_statistic = makemockcategoricalfixed_twotargets
        input_varname1 = "target1"
        codelength1 = categorical_free_encoding(input_statistic, input_varname1)

        input_varname2 = "target2"
        codelength2 = categorical_free_encoding(input_statistic, input_varname2)

        expected_codelength1= 103.72355426179936
        expected_codelength2=  11.802867851390408

        assert expected_codelength1 == pytest.approx(codelength1)
        assert expected_codelength2 == pytest.approx(codelength2)



    def test_1target(self, makemockcategoricalfixed_onetarget):
        input_statistic = makemockcategoricalfixed_onetarget
        input_varname = "target1"
        codelength = categorical_free_encoding(input_statistic, input_varname)

        expected_codelength1= 103.72355426179936

        assert expected_codelength1 == pytest.approx(codelength)

class TestCategoricalFixedEncoding:
    def test_2targets(self, makemockcategoricalfixed_twotargets,generate_inputvalues_two_targets,makemockrulelist):
        input_statistic = makemockcategoricalfixed_twotargets
        data_class = generate_inputvalues_two_targets
        rulelist_class = makemockrulelist
        rulelist = rulelist_class(data_class)
        input_varname1 = "target1"
        codelength1 = categorical_fixed_encoding(rulelist,input_statistic, input_varname1)

        input_varname2 = "target2"
        codelength2 = categorical_fixed_encoding(rulelist,input_statistic, input_varname2)

        expected_codelength1 = 100.0
        expected_codelength2 = 8.079313589591118

        assert expected_codelength1 == pytest.approx(codelength1)
        assert expected_codelength2 == pytest.approx(codelength2)

    def test_1target(self, makemockcategoricalfixed_onetarget,generate_inputvalues_one_target,makemockrulelist):
        input_statistic = makemockcategoricalfixed_onetarget
        data_class  = generate_inputvalues_one_target
        rulelist_class = makemockrulelist
        rulelist =rulelist_class(data_class)
        input_varname = "target1"
        actual_codelength = categorical_fixed_encoding(rulelist, input_statistic, input_varname)

        expected_codelength1 = 100.0

        assert expected_codelength1 == pytest.approx(actual_codelength)

class TestRuleFreeCategorical:
    def test_2targets(self,makemockcategoricalfixed_twotargets,generate_inputvalues_two_targets,makemockrulelist):
        input_statistic = makemockcategoricalfixed_twotargets
        data_class = generate_inputvalues_two_targets
        rulelist_class = makemockrulelist
        rulelist = rulelist_class(data_class)
        actual_codelength = length_rule_free_categorical(rulelist,input_statistic)

        expected_codelength = 103.72355426179936 + 11.802867851390408

        assert expected_codelength == pytest.approx(actual_codelength)

    def test_1targets(self,makemockcategoricalfixed_onetarget,generate_inputvalues_one_target,makemockrulelist):
        input_statistic = makemockcategoricalfixed_onetarget
        data_class = generate_inputvalues_one_target
        rulelist_class = makemockrulelist
        rulelist = rulelist_class(data_class)
        actual_codelength = length_rule_free_categorical(rulelist,input_statistic)

        expected_codelength = 103.72355426179936

        assert expected_codelength == pytest.approx(actual_codelength)


class TestRuleFixedCategorical:
    def test_2targets(self,makemockcategoricalfixed_twotargets,generate_inputvalues_two_targets,makemockrulelist):
        input_statistic = makemockcategoricalfixed_twotargets
        data_class = generate_inputvalues_two_targets
        rulelist_class = makemockrulelist
        rulelist = rulelist_class(data_class)
        actual_codelength = length_rule_fixed_categorical(rulelist,input_statistic)

        expected_codelength = 100 + 8.079313589591118

        assert expected_codelength == pytest.approx(actual_codelength)

    def test_1targets(self,makemockcategoricalfixed_onetarget,generate_inputvalues_one_target,makemockrulelist):
        input_statistic = makemockcategoricalfixed_onetarget
        data_class = generate_inputvalues_one_target
        rulelist_class = makemockrulelist
        rulelist = rulelist_class(data_class)
        actual_codelength = length_rule_fixed_categorical(rulelist,input_statistic)

        expected_codelength = 100

        assert expected_codelength == pytest.approx(actual_codelength)