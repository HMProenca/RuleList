import numpy as np
import pandas as pd
import pytest
from gmpy2 import bit_mask

from rulelist.rulelistmodel.categoricalmodel.categoricaltarget import CategoricalTarget
from rulelist.util.bitset_operations import indexes2bitset


@pytest.fixture
def generate_dataframe_one_target():
    dictoutput = {"target1": np.array(["below50" if i < 50 else "above49" for i in range(100)])}
    input_target_data = pd.DataFrame(data=dictoutput)
    yield input_target_data

@pytest.fixture
def generate_dataframe_two_targets():
    dictoutput = {"target1": np.array(["below50" if i < 50 else "above49" for i in range(100)]),
                  "target2": np.array(["below100" if i < 99 else "above99" for i in range(100)])}
    input_target_data = pd.DataFrame(data=dictoutput)
    yield input_target_data

@pytest.fixture
def generate_inputvalues_explode():
    dictoutput = {"target1": np.array(["below100" for i in range(100)])}
    input_target_data = pd.DataFrame(data=dictoutput)
    yield input_target_data


class TestCategoricalTarget(object):
    def test_onetarget(self,generate_dataframe_one_target):
        input_target_data = generate_dataframe_one_target

        expected_categories = {"target1": np.array(["below50","above49"], dtype=object)}
        expected_bit_array = bit_mask(100)
        expected_number_classes = {"target1": 2}
        expected_bit_arrays_var_class  = {"target1":
                                              {"below50": indexes2bitset(np.arange(50)), "above49": indexes2bitset(np.arange(50,100))}}
        expected_counts  =  {"target1":{"below50": 50, "above49": 50}}
        expected_prob_var_class ={"target1":{"below50": 0.50, "above49": 0.50}}

        output_categoricaltarget = CategoricalTarget(input_target_data)

        assert expected_bit_array ==  output_categoricaltarget.bit_array
        np.testing.assert_array_equal(expected_categories["target1"],output_categoricaltarget.categories["target1"])
        assert expected_number_classes == output_categoricaltarget.number_classes
        assert expected_bit_arrays_var_class == output_categoricaltarget.bit_arrays_var_class
        assert expected_counts == output_categoricaltarget.counts
        assert expected_prob_var_class == output_categoricaltarget.prob_var_class

    def test_twotarget(self,generate_dataframe_two_targets):
        input_target_data = generate_dataframe_two_targets

        expected_categories = {"target1": np.array(["below50","above49"], dtype=object),
                               "target2": np.array(["below100","above99"], dtype=object)}
        expected_bit_array = bit_mask(100)
        expected_number_classes = {"target1": 2,"target2":2}
        expected_bit_arrays_var_class  = {"target1":
                                              {"below50": indexes2bitset(np.arange(50)),
                                               "above49": indexes2bitset(np.arange(50,100))},
                                          "target2":
                                              {"below100": indexes2bitset(np.arange(99)),
                                               "above99": indexes2bitset(np.arange(99, 100))}}

        expected_counts  =  {"target1":{"below50": 50, "above49": 50},
                             "target2":{"below100": 99, "above99": 1}}
        expected_prob_var_class = {"target1":{"below50": 0.50,"above49": 0.50},
                                   "target2":{"below100": 0.99,"above99": 0.01}}

        output_categoricaltarget = CategoricalTarget(input_target_data)

        assert expected_bit_array ==  output_categoricaltarget.bit_array
        np.testing.assert_array_equal(expected_categories["target1"],output_categoricaltarget.categories["target1"])
        np.testing.assert_array_equal(expected_categories["target2"],output_categoricaltarget.categories["target2"])
        assert expected_number_classes == output_categoricaltarget.number_classes
        assert expected_bit_arrays_var_class == output_categoricaltarget.bit_arrays_var_class
        assert expected_counts == output_categoricaltarget.counts
        assert expected_prob_var_class == output_categoricaltarget.prob_var_class

    def test_onlyoneclass_error(self, generate_inputvalues_explode):
        input_target_data = generate_inputvalues_explode

        with pytest.raises(ValueError) as exception_info:  # store the exception
            output_categoricaltarget = CategoricalTarget(input_target_data)

        assert exception_info.match("There is at least one target variable with only one class label. "\
                                    "Please only add targets with 2 or more class labels.")

