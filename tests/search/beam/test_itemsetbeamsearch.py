import numpy as np
import pandas as pd
import pytest

from rulelist.datastructure.data import Data
from rulelist.datastructure.subgroup import Subgroup
from rulelist.rulelistmodel.gaussianmodel.gaussianrulelist import GaussianRuleList
from rulelist.search.beam.beam import Beam
from rulelist.search.beam.itemset_beamsearch import refine_subgroup, find_best_rule


@pytest.fixture
def constant_parameters():
    input_n_cutpoints = 5
    input_discretization = "static"
    input_target_data = "gaussian"
    input_minsupp = 0
    yield input_n_cutpoints, input_discretization, input_target_data, input_minsupp

@pytest.fixture
def generate_input_dataframe_two_target_normal(constant_parameters):
    input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters
    dictinput = {"attribute1": np.arange(100000),
                 "attribute2": np.array(["below1000" if i < 1000 else "above999" for i in range(100000)])}
    input_input_data = pd.DataFrame(data=dictinput)
    dictoutput = {"target1": np.concatenate((np.random.normal(loc=20,scale=3,size=16666),
                                             np.random.normal(loc=100,scale=6,size=83334)), axis=None),
                  "target2": np.concatenate((np.random.normal(loc=10,scale=2,size=16666),
                                             np.random.normal(loc=50,scale=5,size=83334)), axis=None)}
    input_output_data = pd.DataFrame(data=dictoutput)
    data = Data(input_input_data, input_n_cutpoints, input_discretization,
                       input_output_data, input_target_data,input_minsupp)
    yield data


@pytest.fixture
def auxiliar_nominal_candidate(generate_input_dataframe_two_target_normal):
    data = generate_input_dataframe_two_target_normal
    candidate2refine = [data.attributes[1].items[1]]
    yield candidate2refine

@pytest.fixture
def make_rulelist(generate_input_dataframe_two_target_normal):
    data = generate_input_dataframe_two_target_normal
    input_target_model = "gaussian"
    input_task = "discovery"
    input_max_depth = 5
    input_beam_width = 10
    input_max_rules = 10
    input_alpha_gain = 1
    input_minsupp = 0
    input_ruleset = GaussianRuleList(data, input_task, input_max_depth, input_beam_width,input_minsupp,
                                     input_max_rules,input_alpha_gain)
    yield input_ruleset


class TestFindBestRule:
    def test_numeric_candidate(self,generate_input_dataframe_two_target_normal, auxiliar_nominal_candidate,
                               make_rulelist):
        data = generate_input_dataframe_two_target_normal
        input_ruleset = make_rulelist

        subgroup2add = find_best_rule(input_ruleset, data)
        expected_subgroup2add_pattern = [data.attributes[1].items[0]]

        assert expected_subgroup2add_pattern == subgroup2add.pattern

class TestRefineSubgroup:
    def test_numeric_candidate(self,generate_input_dataframe_two_target_normal, auxiliar_nominal_candidate,
                               make_rulelist):
        data = generate_input_dataframe_two_target_normal
        candidate2refine = auxiliar_nominal_candidate
        input_ruleset = make_rulelist
        beam = Beam(beam_width=10)
        subgroup2add = Subgroup()


        beam, subgroup2add = refine_subgroup(input_ruleset, data, candidate2refine, beam, subgroup2add)

        expected_subgroup2add_pattern = candidate2refine + [data.attributes[0].items[0]]

        assert expected_subgroup2add_pattern == subgroup2add.pattern
