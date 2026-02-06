import numpy as np
import pandas as pd
import pytest


from rulelist.datastructure.data import Data


@pytest.fixture
def constant_parameters():
    input_n_cutpoints = 5
    input_discretization = "static"
    input_target_data = "gaussian"
    input_minsupp = 0
    yield input_n_cutpoints, input_discretization, input_target_data, input_minsupp

@pytest.fixture
def generate_input_dataframe_one_target():
    dictinput = {"attribute1": np.arange(100),
                 "attribute2": np.array(["below50" if i < 50 else "above49" for i in range(100)])}
    dictoutput = {"target1": np.arange(100)}

    input_input_data = pd.DataFrame(data=dictinput)
    input_output_data = pd.DataFrame(data=dictoutput)
    yield input_input_data, input_output_data

@pytest.fixture
def generate_input_dataframe_two_target():
    dictinput = {"attribute1": np.arange(100),
                 "attribute2": np.array(["below50" if i < 50 else "above49" for i in range(100)])}
    dictoutput = {"target1": np.arange(100), "target2": np.ones(100)}

    input_input_data = pd.DataFrame(data=dictinput)
    input_output_data = pd.DataFrame(data=dictoutput)
    yield input_input_data, input_output_data



class TestData(object):
    def test_gaussian_onetarget(self,generate_input_dataframe_one_target,constant_parameters):
        input_input_data, input_output_data = generate_input_dataframe_one_target
        input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters

        expected_number_targets = 1
        expected_number_attributes = 2
        expected_number_instances = 100
        expected_attribute_names = {"attribute1", "attribute2"}
        expected_target_names = {"target1"}

        output_data = Data(input_input_data, input_n_cutpoints, input_discretization,
                                            input_output_data, input_target_data,input_minsupp)

        pd.testing.assert_frame_equal(input_input_data,output_data.input_data)
        pd.testing.assert_frame_equal(input_output_data,output_data.target_data)
        assert expected_number_attributes ==  output_data.number_attributes
        assert expected_number_attributes ==  len(output_data.attributes)
        assert expected_number_targets ==  output_data.number_targets
        assert expected_number_instances ==  output_data.number_instances
        assert expected_attribute_names ==  output_data.attribute_names
        assert expected_target_names ==  output_data.target_names

    def test_gaussian_twotargets(self,generate_input_dataframe_two_target,constant_parameters):
        input_input_data, input_output_data = generate_input_dataframe_two_target
        input_n_cutpoints, input_discretization, input_target_data,input_minsupp = constant_parameters

        expected_number_targets = 2
        expected_number_attributes = 2
        expected_number_instances = 100
        expected_attribute_names = {"attribute1", "attribute2"}
        expected_target_names = {"target1","target2"}

        output_data = Data(input_input_data, input_n_cutpoints, input_discretization,
                                            input_output_data, input_target_data,input_minsupp)

        pd.testing.assert_frame_equal(input_input_data,output_data.input_data)
        pd.testing.assert_frame_equal(input_output_data,output_data.target_data)
        assert expected_number_attributes ==  output_data.number_attributes
        assert expected_number_attributes ==  len(output_data.attributes)
        assert expected_number_targets ==  output_data.number_targets
        assert expected_number_instances ==  output_data.number_instances
        assert expected_attribute_names ==  output_data.attribute_names
        assert expected_target_names ==  output_data.target_names

        @pytest.mark.xfail
        def test_name_not_present(self):
            pass

        @pytest.mark.xfail
        def test_category_not_present(self):
            pass

        @pytest.mark.xfail
        def test_receives_series(self):
            pass