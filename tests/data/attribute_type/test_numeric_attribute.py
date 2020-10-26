import numpy as np
import pandas as pd
import pytest

from rulelist.datastructure.attribute.numeric_attribute import activation_numeric, find_cutpoints, NumericAttribute, \
    create_item


class TestNumericAttribute(object):
    def test_normal(self):
        dictdata = {"column1" : np.arange(100),"column2" : np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        input_name = "column1"
        input_max_operators = 2
        input_n_cutpoints = 3
        input_minsupp = 0
        input_discretization = "static"
        expected_number_items = 9 # 3 with <, 3 with >, and 3 with <>
        expected_cardinality_operator = {1: 6, 2: 3}
        expected_n_cutpoints = 3

        output_attribute = NumericAttribute(input_name, test_dataframe[input_name], input_max_operators,input_minsupp,
                                            input_n_cutpoints, input_discretization)
        actual_number_items= len(output_attribute.items)
        actual_cardinality_operator = output_attribute.cardinality_operator
        actual_n_cutpoints = output_attribute.n_cutpoints

        pd.testing.assert_series_equal(output_attribute.values, test_dataframe[input_name], check_exact=True)
        assert expected_number_items == actual_number_items
        assert expected_cardinality_operator == actual_cardinality_operator
        assert expected_n_cutpoints == actual_n_cutpoints


    def test_onlyonevalue(self):
        dictdata = {"column1": np.arange(100), "column2": np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        input_name = "column2"
        input_max_operators = 2
        input_n_cutpoints = 5
        input_discretization = "static"
        input_minsupp = 0

        expected_number_items = 1  # 3 with <, 3 with >, and 3 with <>
        expected_cardinality_operator = {1:1, 2: 0}
        expected_n_cutpoints = 1

        output_attribute = NumericAttribute(input_name, test_dataframe[input_name], input_max_operators,input_minsupp,
                                            input_n_cutpoints, input_discretization)
        actual_number_items = len(output_attribute.items)
        actual_cardinality_operator = output_attribute.cardinality_operator
        actual_n_cutpoints = output_attribute.n_cutpoints

        pd.testing.assert_series_equal(output_attribute.values, test_dataframe[input_name], check_exact=True)
        assert expected_number_items == actual_number_items
        assert expected_cardinality_operator == actual_cardinality_operator
        assert expected_n_cutpoints == actual_n_cutpoints

class TestCreateItem(object):
    def test_less_cutpoints_than_points(self):
        dictdata = {"column1": np.arange(100), "column2": np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        input_indexes = np.array([3, 4, 5], dtype = np.int32)
        input_name = "column1"
        input_min_val = 3
        input_max_val = 6
        input_description = " 3 <= column1 < 6"
        input_number_operations = 2
        output_item = create_item(input_indexes, input_name, input_min_val, input_max_val, input_description,
                           input_number_operations)

        expected_vector = pd.Series(name= "column1", data = [True if i in input_indexes else False for i in range(100)])
        actual_vector = output_item.activation_function(test_dataframe)
        pd.testing.assert_series_equal(actual_vector, expected_vector, check_exact=True)


class TestActivationNumeric(object):
    def test_left_interval(self):
        dictdata = {"column1" : np.arange(100),"column2" : np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        attribute_test = "column1"
        minval = np.NINF
        maxval = 50
        expected_vector = pd.Series(name= "column1", data = [True if i < maxval else False for i in range(100)])
        test_vector = activation_numeric(df = test_dataframe, attribute_name = attribute_test, minval = minval, maxval = maxval)
        pd.testing.assert_series_equal(test_vector, expected_vector, check_exact=True)

    def test_right_interval(self):
        dictdata = {"column1" : np.arange(100),"column2" : np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        attribute_test = "column1"
        minval = 50
        maxval = np.inf
        expected_vector = pd.Series(name= "column1", data = [True if i >= minval else False for i in range(100)])
        test_vector = activation_numeric(df = test_dataframe, attribute_name = attribute_test, minval = minval, maxval = maxval)
        pd.testing.assert_series_equal(test_vector, expected_vector, check_exact=True)

    def test_binary_interval(self):
        dictdata = {"column1" : np.arange(100),"column2" : np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        attribute_test = "column1"
        minval = 50
        maxval = 20
        expected_vector = pd.Series(name= "column1", data = [True if i >= minval and i < maxval else False for i in range(100)])
        test_vector = activation_numeric(df = test_dataframe, attribute_name = attribute_test, minval = minval, maxval = maxval)
        pd.testing.assert_series_equal(test_vector, expected_vector, check_exact=True)

    def test_allfalse(self):
        dictdata = {"column1" : np.arange(100),"column2" : np.ones(100)}
        test_dataframe = pd.DataFrame(data=dictdata)
        attribute_test = "column1"
        minval = 100
        maxval = 200
        expected_vector = pd.Series(name= "column1", data = [True if i >= minval and i < maxval else False for i in range(100)])
        test_vector = activation_numeric(df = test_dataframe, attribute_name = attribute_test, minval = minval, maxval = maxval)
        pd.testing.assert_series_equal(test_vector, expected_vector, check_exact=True)

    @pytest.mark.xfail
    def test_name_not_present(self):
        pass

    @pytest.mark.xfail
    def test_receives_series(self):
        pass

class TestFindCutPoints(object):
    def test_less_cutpoints_than_points(self):
        test_input = np.array([1, 2, 3, 4, 5])
        n_cutpoints = 3
        expected_quantiles = np.array([2., 3., 4.])
        expected_real_ncutpoints = 3
        actual_quantiles, actual_real_ncutpoints = find_cutpoints(test_input, n_cutpoints)
        assert expected_quantiles == pytest.approx(actual_quantiles)
        assert expected_real_ncutpoints == actual_real_ncutpoints

    def test_exactcutpoints(self):
        test_input = np.array([1, 2, 3, 4, 5])
        n_cutpoints = 5
        expected_quantiles = np.array([1.5, 2.5, 3. , 3.5, 4.5])
        expected_real_ncutpoints = 5
        actual_quantiles, actual_real_ncutpoints = find_cutpoints(test_input, n_cutpoints)
        assert expected_quantiles == pytest.approx(actual_quantiles)
        assert expected_real_ncutpoints == actual_real_ncutpoints

    def test_more_cutpoints_than_points(self):
        test_input = np.array([1, 2, 3, 4, 5])
        n_cutpoints = 20
        expected_quantiles = np.array([1.5, 2.5, 3. , 3.5, 4.5])
        expected_real_ncutpoints = 5
        actual_quantiles, actual_real_ncutpoints = find_cutpoints(test_input, n_cutpoints)
        assert expected_quantiles == pytest.approx(actual_quantiles)
        assert expected_real_ncutpoints == actual_real_ncutpoints


    def test_allthesamevalue(self):
        test_input = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
        n_cutpoints = 5
        expected_quantiles = np.array([1.])
        expected_real_ncutpoints = 1
        actual_quantiles, actual_real_ncutpoints = find_cutpoints(test_input, n_cutpoints)
        assert expected_quantiles == pytest.approx(actual_quantiles)
        assert expected_real_ncutpoints == actual_real_ncutpoints

