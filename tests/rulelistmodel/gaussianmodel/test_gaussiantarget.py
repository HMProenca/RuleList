import numpy as np
import pandas as pd
from gmpy2 import bit_mask

from rulelist.rulelistmodel.gaussianmodel.gaussiantarget import GaussianTargets


class TestGaussianTargets(object):
    def test_onetarget(self):
        dictoutput = {"target1": np.arange(100)}
        input_target_data = pd.DataFrame(data=dictoutput)

        expected_number_targets = 1
        expected_bit_array = bit_mask(100)
        expected_mean_vector = np.array([49.5])
        expected_variance_vector  = np.var([*range(100)])

        output_gaussiantargets= GaussianTargets(input_target_data)

        assert expected_bit_array ==  output_gaussiantargets.bit_array
        assert expected_number_targets == len(output_gaussiantargets.mean)
        assert expected_number_targets == len(output_gaussiantargets.variance)

        np.testing.assert_array_equal(expected_mean_vector, output_gaussiantargets.mean)
        np.testing.assert_array_equal(expected_variance_vector, output_gaussiantargets.variance)

    def test_twotargets(self):
        dictoutput = {"target1": np.arange(100), "target2": np.ones(100)}
        input_target_data = pd.DataFrame(data=dictoutput)

        expected_number_targets = 2
        expected_bit_array = bit_mask(100)
        expected_mean_vector = np.array([49.5,1])
        expected_variance_vector  = [np.var([*range(100)]),0]

        output_gaussiantargets= GaussianTargets(input_target_data)

        assert expected_bit_array ==  output_gaussiantargets.bit_array
        assert expected_number_targets == len(output_gaussiantargets.mean)
        assert expected_number_targets == len(output_gaussiantargets.variance)
        np.testing.assert_array_equal(expected_mean_vector, output_gaussiantargets.mean)
        np.testing.assert_array_equal(expected_variance_vector, output_gaussiantargets.variance)

    def test_onlyzeros(self):
        dictoutput = {"target1": np.zeros(100)}
        input_target_data = pd.DataFrame(data=dictoutput)

        expected_number_targets = 1
        expected_bit_array = bit_mask(100)
        expected_mean_vector = np.array([0])
        expected_variance_vector = np.array([0])

        output_gaussiantargets = GaussianTargets(input_target_data)

        assert expected_bit_array == output_gaussiantargets.bit_array
        assert expected_number_targets == len(output_gaussiantargets.mean)
        assert expected_number_targets == len(output_gaussiantargets.variance)
        np.testing.assert_array_equal(expected_mean_vector, output_gaussiantargets.mean)
        np.testing.assert_array_equal(expected_variance_vector, output_gaussiantargets.variance)