import numpy as np
import pandas as pd
import pytest

from rulelist.search.iterative_rule_search import _fit_rulelist


@pytest.fixture
def constant_parameters():
    input_n_cutpoints = 5
    input_discretization = "static"
    input_target_model = "gaussian"
    input_max_depth = 5
    input_beam_width = 10
    input_iterative_beam_width = 1
    input_task = "discovery"
    input_max_rules= 10
    input_alpha_gain = 1

    yield input_n_cutpoints, input_discretization, input_target_model,input_max_depth, input_beam_width,\
            input_iterative_beam_width, input_task, input_max_rules, input_alpha_gain

@pytest.fixture
def generate_input_dataframe_two_target_normal(constant_parameters):
    input_n_cutpoints, input_discretization, input_target_model, input_max_depth, input_beam_width, \
            input_iterative_beam_width, input_task, input_max_rules, input_alpha_gain = constant_parameters
    dictinput = {"attribute1": np.arange(100000),
                 "attribute2": np.array(["below1000" if i < 1000 else "above999" for i in range(100000)])}
    input_input_data = pd.DataFrame(data=dictinput)
    dictoutput = {"target1": np.concatenate((np.random.normal(loc=20,scale=3,size=16666),
                                             np.random.normal(loc=100,scale=6,size=83334)), axis=None),
                  "target2": np.concatenate((np.random.normal(loc=10,scale=2,size=16666),
                                             np.random.normal(loc=50,scale=5,size=83334)), axis=None)}
    input_output_data = pd.DataFrame(data=dictoutput)
    yield input_input_data, input_output_data

class TestFitRuleList:
    def test_start(self,constant_parameters,generate_input_dataframe_two_target_normal):
        input_n_cutpoints, input_discretization, input_target_model, input_max_depth, input_beam_width, \
            input_iterative_beam_width, input_task, input_max_rules, input_alpha_gain  = constant_parameters
        input_input_data, input_output_data = generate_input_dataframe_two_target_normal

        output_rulelist = _fit_rulelist(input_input_data, input_output_data, input_target_model, input_max_depth,
                                        input_beam_width, input_iterative_beam_width,input_n_cutpoints, input_task,
                                        input_discretization, input_max_rules, input_alpha_gain)
