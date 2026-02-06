from math import log2

import numpy as np
import pandas as pd
import pytest

from rulelist.datastructure.attribute import NumericAttribute, NominalAttribute
from rulelist.mdl.mdl_base_codes import universal_code_integers_maximum
from rulelist.rulelistmodel.model_encoding import compute_item_length


@pytest.fixture
def auxiliar_numericattribute():
    dictdata = {"column1": np.arange(100), "column2": np.ones(100)}
    test_dataframe = pd.DataFrame(data=dictdata)
    input_name = "column1"
    input_max_operators = 2
    input_n_cutpoints = 3
    input_minsupp = 0
    input_discretization = "static"
    numericattribute = NumericAttribute(input_name, test_dataframe[input_name], input_max_operators,input_minsupp,
                                    input_n_cutpoints, input_discretization)
    return numericattribute

@pytest.fixture
def auxiliar_nominalattribute():
    dictdata = {"column1": np.array(["below50" if i < 50 else "above49" for i in range(100)]),
                "column2": np.ones(100)}
    test_dataframe = pd.DataFrame(data=dictdata)
    input_name = "column1"
    input_max_operators = 1
    input_minsupp = 0
    nominalattribute = NominalAttribute(input_name, test_dataframe[input_name], input_max_operators,input_minsupp)
    return nominalattribute


class TestComputeItemLength:
    def test_numericattribute(self,auxiliar_numericattribute):
        numericattribute = auxiliar_numericattribute
        expected_length_item_1_operator = log2(6) + universal_code_integers_maximum(1, 2)
        expected_length_item_2_operator = log2(3) + universal_code_integers_maximum(2, 2)

        expected_output = [("column1",1,expected_length_item_1_operator),
                           ("column1", 2, expected_length_item_2_operator)]
        output = [*compute_item_length(numericattribute)]
        assert expected_output[0][0] == output[0][0]
        assert expected_output[0][1] == output[0][1]
        assert expected_output[0][2] == pytest.approx(output[0][2])
        assert expected_output[1][0] == output[1][0]
        assert expected_output[1][1] == output[1][1]
        assert expected_output[1][2] == pytest.approx(output[1][2])

    def test_nominalattribute(self,auxiliar_nominalattribute):
        nominalattribute = auxiliar_nominalattribute
        expected_length_item_1_operator = log2(2) + universal_code_integers_maximum(1, 1)

        expected_output = [("column1",1,expected_length_item_1_operator)]
        output = [*compute_item_length(nominalattribute)]
        assert expected_output[0][0] == output[0][0]
        assert expected_output[0][1] == output[0][1]
        assert expected_output[0][2] == pytest.approx(output[0][2])