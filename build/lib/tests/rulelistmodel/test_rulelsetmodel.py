import numpy as np
import pandas as pd
import pytest
from gmpy2 import mpz, bit_mask

from rulelist.datastructure.data import Data
from rulelist.rulelistmodel.rulesetmodel import RuleSetModel


@pytest.fixture
def constant_parameters():
    input_n_cutpoints = 5
    input_discretization = "static"
    input_target_data = "gaussian"
    input_minsupp = 0
    dictinput = {"attribute1": np.arange(100),
                 "attribute2": np.array(["below50" if i < 50 else "above49" for i in range(100)])}
    input_input_data = pd.DataFrame(data=dictinput)
    dictoutput = {"target1": np.arange(100), "target2": np.ones(100)}
    input_output_data = pd.DataFrame(data=dictoutput)
    yield input_input_data, input_output_data, input_n_cutpoints, input_discretization, input_target_data,input_minsupp

@pytest.fixture
def generate_input_dataframe_two_target(constant_parameters):
    input_input_data, input_output_data, input_n_cutpoints, input_discretization, input_target_data ,input_minsupp\
        = constant_parameters
    data = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    yield data

class TestRuleSetModel:
    def test_initialization(self, generate_input_dataframe_two_target):
        data = generate_input_dataframe_two_target
        input_task = "discovery"
        input_target_model = "gaussian"
        input_max_depth = 5
        input_beam_width = 10
        input_minsupp = 0
        input_max_rules = 10
        input_alpha_gain = 1

        expected_task = input_task
        expected_target_model = input_target_model
        expected_alpha = 1
        expected_beam_width = input_beam_width
        expected_max_depth = min(input_max_depth,data.number_attributes)
        expected_max_rules = input_max_rules
        expected_bitset_covered = mpz()
        expected_support_covered = 0
        expected_bitset_uncovered = bit_mask(data.number_instances)
        expected_support_uncovered = data.number_instances
        expected_default_rule_statistics_usage = data.number_instances
        expected_length_data = None
        expected_length_original = None
        expected_length_defaultrule = None
        expected_length_ratio = 1.0

        expected_subgroups = []
        expected_length_model = 0

        output_ruleset = RuleSetModel(data,input_task, input_max_depth,input_beam_width,input_minsupp,
                                      input_max_rules,input_alpha_gain)

        assert expected_task == output_ruleset.task
        assert expected_target_model == output_ruleset.target_model
        assert expected_alpha  == output_ruleset.alpha_gain
        assert expected_beam_width == output_ruleset.beam_width
        assert expected_max_depth == output_ruleset.max_depth
        assert expected_max_rules == output_ruleset.max_rules
        assert expected_bitset_covered == output_ruleset.bitset_covered
        assert expected_support_covered == output_ruleset.support_covered
        assert expected_bitset_uncovered == output_ruleset.bitset_uncovered
        assert expected_support_uncovered == output_ruleset.support_uncovered
        assert expected_subgroups == output_ruleset.subgroups
        assert expected_length_model == output_ruleset.length_model
        assert expected_default_rule_statistics_usage == output_ruleset.default_rule_statistics.usage
        assert expected_length_data == output_ruleset.length_data
        assert expected_length_original == output_ruleset.length_original
        assert expected_length_defaultrule == output_ruleset.length_defaultrule
        assert expected_length_ratio == output_ruleset.length_ratio