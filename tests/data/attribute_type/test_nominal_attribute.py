import numpy as np
import pandas as pd
import pytest

from rulelist.datastructure.attribute.nominal_attribute import activation_nominal, NominalAttribute


class TestNominalAttribute(object):
    def test_normal(self):
        dictdata = {"column1" : np.array(["below50" if i < 50 else "above49" for i in range(100)]),
                    "column2" : np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        input_name = "column1"
        input_max_operators = 1
        input_minsupp = 0
        expected_number_items = 2
        expected_cardinality_operator = {1: 2}

        output_attribute = NominalAttribute(input_name, test_dataframe[input_name], input_max_operators,input_minsupp)
        actual_number_items= len(output_attribute.items)
        actual_cardinality_operator = output_attribute.cardinality_operator

        pd.testing.assert_series_equal(output_attribute.values, test_dataframe[input_name])
        assert expected_number_items == actual_number_items
        assert expected_cardinality_operator == actual_cardinality_operator

    def test_onlyonevalue(self):
        dictdata = {"column1" : np.array(["below100" for i in range(100)]),
                    "column2" : np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        input_name = "column1"
        input_max_operators = 1
        input_minsupp = 0

        expected_number_items = 1
        expected_cardinality_operator = {1: 1}
        expected_n_cutpoints = 3

        output_attribute = NominalAttribute(input_name, test_dataframe[input_name], input_max_operators,input_minsupp)
        actual_number_items= len(output_attribute.items)
        actual_cardinality_operator = output_attribute.cardinality_operator

        pd.testing.assert_series_equal(output_attribute.values, test_dataframe[input_name])
        assert expected_number_items == actual_number_items
        assert expected_cardinality_operator == actual_cardinality_operator

class TestActivationNominal(object):
    def test_left_interval(self):
        dictdata = {"column1" : np.array(["below50" if i < 50 else "above49" for i in range(100)]),
                    "column2" : np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        input_attribute_name = "column1"
        input_category = "below50"

        expected_vector = pd.Series(name= "column1", data = [True if i < 50 else False for i in range(100)])
        actual_vector = activation_nominal(test_dataframe,input_attribute_name,input_category)
        pd.testing.assert_series_equal(actual_vector, expected_vector, check_exact=True)

    def test_right_interval(self):
        dictdata = {"column1": np.array(["below50" if i < 50 else "above49" for i in range(100)]),
                    "column2": np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        input_attribute_name = "column1"
        input_category = "above49"
        expected_vector = pd.Series(name="column1", data=[True if i > 49 else False for i in range(100)])
        actual_vector = activation_nominal(test_dataframe, input_attribute_name, input_category)
        pd.testing.assert_series_equal(actual_vector, expected_vector, check_exact=True)

    @pytest.mark.xfail
    def test_name_not_present(self):
        pass

    @pytest.mark.xfail
    def test_category_not_present(self):
        pass

    @pytest.mark.xfail
    def test_receives_series(self):
        pass