from math import log2, exp, pi

import numpy as np
import pandas as pd
import pytest
from gmpy2 import mpz, bit_mask, popcount

from rulelist.datastructure.data import Data
from rulelist.datastructure.subgroup import Subgroup
from rulelist.mdl.mdl_base_codes import universal_code_integers, uniform_combination_code, \
    universal_code_integers_maximum, \
    uniform_code
from rulelist.rulelistmodel.gaussianmodel.gaussianrulelist import GaussianRuleList
from rulelist.rulelistmodel.gaussianmodel.gaussianstatistic import GaussianFixedStatistic, GaussianFreeStatistic
from rulelist.util.bitset_operations import indexes2bitset


@pytest.fixture
def search_parameters():
    input_target_model = "gaussian"
    input_max_depth = 5
    input_beam_width = 10
    input_minsupp = 0
    input_max_rules = 10
    input_alpha_gain = 1
    yield input_target_model, input_max_depth, input_beam_width,input_minsupp, input_max_rules, input_alpha_gain

@pytest.fixture
def constant_parameters():
    input_n_cutpoints = 5
    input_discretization = "static"
    input_target_data = "gaussian"
    input_minsupp = 0
    yield input_n_cutpoints, input_discretization, input_target_data, input_minsupp

@pytest.fixture
def generate_input_dataframe_two_target_variancezero(constant_parameters):
    input_n_cutpoints, input_discretization, input_target_data, input_minsupp = constant_parameters
    dictinput = {"attribute1": np.arange(100),
                 "attribute2": np.array(["below50" if i < 50 else "above49" for i in range(100)])}
    input_input_data = pd.DataFrame(data=dictinput)
    dictoutput = {"target1": np.arange(100), "target2": np.ones(100)}
    input_output_data = pd.DataFrame(data=dictoutput)
    data = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    yield data

@pytest.fixture
def generate_input_dataframe_two_target_normal(constant_parameters):
    input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters
    dictinput = {"attribute1": np.arange(100000),
                 "attribute2": np.array(["below1000" if i < 1000 else "above999" for i in range(100000)])}
    input_input_data = pd.DataFrame(data=dictinput)
    dictoutput = {"target1": np.random.normal(loc=20,scale=3,size=100000),
                  "target2": np.random.normal(loc=10,scale=1,size=100000)}
    input_output_data = pd.DataFrame(data=dictoutput)
    data = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    yield data

@pytest.fixture
def generate_input_dataframe_one_target(constant_parameters):
    input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters
    dictinput = {"attribute1": np.arange(100000),
                 "attribute2": np.array(["below1000" if i < 1000 else "above999" for i in range(100000)])}
    input_input_data = pd.DataFrame(data=dictinput)
    dictoutput = {"target1": np.random.normal(loc=1,scale=1,size=100000)}
    input_output_data = pd.DataFrame(data=dictoutput)
    data = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    yield data

@pytest.fixture
def generate_subgroup_oneitem_numeric(generate_input_dataframe_two_target_normal):
    data = generate_input_dataframe_two_target_normal
    candidate = [data.attributes[0].items[0]]
    bitarray_aux = candidate[0].bitarray
    statistic_candidate = GaussianFreeStatistic(data)
    statistic_candidate.replace_stats(data,bitarray_aux)
    subgroup2add = Subgroup()
    subgroup2add.update(candidate, statistic_candidate, score=1, gain_data=1, gain_model=1)
    yield subgroup2add

@pytest.fixture
def generate_subgroup_2item_numeric_and_nominal(generate_input_dataframe_two_target_normal):
    data = generate_input_dataframe_two_target_normal
    candidate = [data.attributes[0].items[0], data.attributes[1].items[0]]
    statistic_candidate = GaussianFreeStatistic(data)
    statistic_candidate.replace_stats(data,candidate[0].bitarray & candidate[1].bitarray)
    subgroup2add = Subgroup()
    subgroup2add.update(candidate, statistic_candidate, score=1, gain_data=1, gain_model=1)
    yield subgroup2add

@pytest.fixture
def generate_subgroup_2subgroups(generate_input_dataframe_two_target_normal):
    data = generate_input_dataframe_two_target_normal
    candidate1 = [data.attributes[0].items[0]]
    statistic_candidate1 = GaussianFreeStatistic(data)
    statistic_candidate1.replace_stats(data, candidate1[0].bitarray)

    subgroup2add1 = Subgroup()
    subgroup2add1.update(candidate1, statistic_candidate1, score=1, gain_data=1, gain_model=1)

    candidate2 = [data.attributes[1].items[1]]
    statistic_candidate2 = GaussianFreeStatistic(data)
    statistic_candidate2.replace_stats(data, candidate2[0].bitarray &~ candidate1[0].bitarray)

    subgroup2add2 = Subgroup()
    subgroup2add2.update(candidate2, statistic_candidate2, score=1, gain_data=1, gain_model=1)
    yield subgroup2add1, subgroup2add2

class TestGaussianRuleList:
    def test_initialization_lengthinfinity(self, generate_input_dataframe_two_target_variancezero,search_parameters):
        # the length of the datastructure is infinity
        data = generate_input_dataframe_two_target_variancezero
        input_target_model, input_max_depth, input_beam_width, input_minsupp, input_max_rules, input_alpha_gain = search_parameters
        input_task = "discovery"

        expected_statistic_type = GaussianFixedStatistic
        expected_length_data = np.inf
        expected_length_original = np.inf
        expected_length_defaultrule = np.inf

        output_ruleset = GaussianRuleList(data,input_task, input_max_depth,input_beam_width,input_minsupp,
                                          input_max_rules,input_alpha_gain)

        assert isinstance(output_ruleset.default_rule_statistics, expected_statistic_type)
        assert expected_length_data == output_ruleset.length_data
        assert expected_length_original == output_ruleset.length_original
        assert expected_length_defaultrule == output_ruleset.length_defaultrule

    def test_initialization_1target_discovery(self,search_parameters, generate_input_dataframe_one_target):
        # the length of the datastructure is infinity
        data = generate_input_dataframe_one_target
        input_target_model, input_max_depth, input_beam_width,input_minsupp, input_max_rules, input_alpha_gain = search_parameters
        input_task = "discovery"

        expected_statistic_type = GaussianFixedStatistic
        expected_length_data_minimum = 0.5*100000 + 0.5*100000

        output_ruleset = GaussianRuleList(data,input_task, input_max_depth,input_beam_width,input_minsupp,
                                          input_max_rules,input_alpha_gain)

        assert isinstance(output_ruleset.default_rule_statistics, expected_statistic_type)
        assert expected_length_data_minimum < output_ruleset.length_defaultrule


    def test_initialization_2targets_discovery(self,search_parameters, generate_input_dataframe_two_target_normal):
        data = generate_input_dataframe_two_target_normal
        input_target_model, input_max_depth, input_beam_width, input_minsupp, input_max_rules, input_alpha_gain = search_parameters
        input_task = "discovery"

        expected_statistic_type = GaussianFixedStatistic
        expected_length_data_minimum = 0.5*100000*(log2(2*pi*1*exp(1)))+0.5*100000*(log2(2*pi*1*exp(1)))

        output_ruleset = GaussianRuleList(data,input_task, input_max_depth,input_beam_width,input_minsupp,
                                          input_max_rules,input_alpha_gain)

        assert isinstance(output_ruleset.default_rule_statistics, expected_statistic_type)
        assert expected_length_data_minimum < output_ruleset.length_defaultrule

    def test_initialization_1target_prediction(self,search_parameters, generate_input_dataframe_one_target):
        # the length of the datastructure is infinity
        data = generate_input_dataframe_one_target
        input_target_model, input_max_depth, input_beam_width, input_minsupp, input_max_rules, input_alpha_gain = search_parameters
        input_task = "prediction"


        expected_statistic_type = GaussianFreeStatistic
        expected_length_data_minimum = 0.5*100000 + 0.5*100000

        output_ruleset = GaussianRuleList(data,input_task, input_max_depth,input_beam_width,input_minsupp,
                                          input_max_rules,input_alpha_gain)

        assert isinstance(output_ruleset.default_rule_statistics, expected_statistic_type)
        assert expected_length_data_minimum < output_ruleset.length_defaultrule


    def test_initialization_2targets_prediction(self,search_parameters, generate_input_dataframe_two_target_normal):
        data = generate_input_dataframe_two_target_normal
        input_target_model, input_max_depth, input_beam_width, input_minsupp, input_max_rules, input_alpha_gain = search_parameters
        input_task = "prediction"

        expected_statistic_type = GaussianFreeStatistic
        expected_length_data_minimum = 0.5*100000*(log2(2*pi*1*exp(1)))+0.5*100000*(log2(2*pi*1*exp(1)))

        output_ruleset = GaussianRuleList(data,input_task, input_max_depth,input_beam_width,input_minsupp,
                                          input_max_rules,input_alpha_gain)

        assert isinstance(output_ruleset.default_rule_statistics, expected_statistic_type)
        assert expected_length_data_minimum < output_ruleset.length_defaultrule

    def test_add_rule_itemnumeric(self,search_parameters, generate_input_dataframe_two_target_normal,
                                  generate_subgroup_oneitem_numeric):
        data = generate_input_dataframe_two_target_normal
        input_target_model, input_max_depth, input_beam_width, input_minsupp, input_max_rules, input_alpha_gain = search_parameters
        subgroup2add = generate_subgroup_oneitem_numeric
        input_task = "discovery"

        output_ruleset = GaussianRuleList(data,input_task, input_max_depth,input_beam_width,input_minsupp,
                                          input_max_rules,input_alpha_gain)
        output_ruleset.add_rule(subgroup2add,data)


        expected_number_instances = data.number_instances
        expected_bitset_uncovered = indexes2bitset([i for i in range(expected_number_instances) if i >16666])
        expected_bitset_covered =  indexes2bitset([i for i in range(expected_number_instances) if i <16666+1])
        expected_number_rules = 1
        expected_length_model = universal_code_integers(1) + universal_code_integers(1)+\
                                uniform_combination_code(1, 2) + universal_code_integers_maximum(1,2)+ \
                                uniform_code(10)


        actual_numberinstances1 = popcount(output_ruleset.subgroups[0].bitarray) +\
                                 popcount(output_ruleset.bitset_uncovered)
        actual_numberinstances2 = output_ruleset.support_covered + output_ruleset.support_uncovered
        actual_numberinstances3 = popcount(output_ruleset.bitset_covered) +\
                                 popcount(output_ruleset.bitset_uncovered)
        actual_numberinstances4 = output_ruleset.subgroups[0].usage + output_ruleset.default_rule_statistics.usage


        assert expected_number_instances == actual_numberinstances1
        assert expected_number_instances == actual_numberinstances2
        assert expected_number_instances == actual_numberinstances3
        assert expected_number_instances == actual_numberinstances4
        assert expected_bitset_uncovered == output_ruleset.bitset_uncovered
        assert expected_bitset_covered == output_ruleset.bitset_covered
        assert expected_number_rules == output_ruleset.number_rules
        assert expected_length_model == pytest.approx(output_ruleset.length_model)

    def test_add_rule_2items(self, search_parameters, generate_input_dataframe_two_target_normal,
                             generate_subgroup_2item_numeric_and_nominal):
        data = generate_input_dataframe_two_target_normal
        input_target_model, input_max_depth, input_beam_width, input_minsupp, input_max_rules, input_alpha_gain = search_parameters
        subgroup2add = generate_subgroup_2item_numeric_and_nominal
        input_task = "discovery"

        output_ruleset = GaussianRuleList(data, input_task, input_max_depth, input_beam_width,input_minsupp,
                                          input_max_rules,input_alpha_gain)
        output_ruleset.add_rule(subgroup2add, data)

        expected_number_instances = data.number_instances
        expected_bitset_uncovered = indexes2bitset([i for i in range(expected_number_instances) if i > 1000])
        expected_bitset_covered = indexes2bitset([i for i in range(expected_number_instances) if i < 1000 + 1])
        expected_number_rules = 1
        expected_length_model = universal_code_integers(1) + universal_code_integers(2) +\
                                uniform_combination_code(2, 2) +\
                                universal_code_integers_maximum(1, 2) + uniform_code(10)+ \
                                universal_code_integers_maximum(1, 1) + uniform_code(2)

        actual_numberinstances1 = popcount(output_ruleset.subgroups[0].bitarray) + \
                                  popcount(output_ruleset.bitset_uncovered)
        actual_numberinstances2 = output_ruleset.support_covered + output_ruleset.support_uncovered
        actual_numberinstances3 = popcount(output_ruleset.bitset_covered) + \
                                  popcount(output_ruleset.bitset_uncovered)
        actual_numberinstances4 = output_ruleset.subgroups[0].usage + output_ruleset.default_rule_statistics.usage

        assert expected_number_instances == actual_numberinstances1
        assert expected_number_instances == actual_numberinstances2
        assert expected_number_instances == actual_numberinstances3
        assert expected_number_instances == actual_numberinstances4
        assert expected_bitset_uncovered == output_ruleset.bitset_uncovered
        assert expected_bitset_covered == output_ruleset.bitset_covered
        assert expected_number_rules == output_ruleset.number_rules
        assert expected_length_model == pytest.approx(output_ruleset.length_model)

    def test_add_rule_2items(self, search_parameters, generate_input_dataframe_two_target_normal,
                             generate_subgroup_2subgroups):
        data = generate_input_dataframe_two_target_normal
        input_target_model, input_max_depth, input_beam_width, input_minsupp, input_max_rules, input_alpha_gain = search_parameters
        subgroup2add1,subgroup2add2  = generate_subgroup_2subgroups
        input_task = "discovery"

        output_ruleset = GaussianRuleList(data, input_task, input_max_depth, input_beam_width, input_minsupp,
                                          input_max_rules,input_alpha_gain)
        output_ruleset.add_rule(subgroup2add1, data)
        output_ruleset.add_rule(subgroup2add2, data)

        expected_number_instances = data.number_instances
        expected_bitset_uncovered = mpz()
        expected_bitset_covered = bit_mask(100000)
        expected_number_rules = 2
        expected_length_model = universal_code_integers(2) + \
                                universal_code_integers(1) +uniform_combination_code(1, 2) +\
                                universal_code_integers_maximum(1, 2) + uniform_code(10)+ \
                                universal_code_integers(1) + uniform_combination_code(1, 2) + \
                                universal_code_integers_maximum(1, 1) + uniform_code(2)

        actual_numberinstances1 = popcount(output_ruleset.subgroups[0].bitarray) + \
                                  popcount(output_ruleset.subgroups[1].bitarray &~ output_ruleset.subgroups[0].bitarray) + \
                                  popcount(output_ruleset.bitset_uncovered)
        actual_numberinstances2 = output_ruleset.support_covered + output_ruleset.support_uncovered
        actual_numberinstances3 = popcount(output_ruleset.bitset_covered) + \
                                  popcount(output_ruleset.bitset_uncovered)
        actual_numberinstances4 = output_ruleset.subgroups[0].usage + output_ruleset.subgroups[1].usage +\
                                  output_ruleset.default_rule_statistics.usage

        assert expected_number_instances == actual_numberinstances1
        assert expected_number_instances == actual_numberinstances2
        assert expected_number_instances == actual_numberinstances3
        assert expected_number_instances == actual_numberinstances4
        assert expected_bitset_uncovered == output_ruleset.bitset_uncovered
        assert expected_bitset_covered == output_ruleset.bitset_covered
        assert expected_number_rules == output_ruleset.number_rules
        assert expected_length_model == pytest.approx(output_ruleset.length_model)




