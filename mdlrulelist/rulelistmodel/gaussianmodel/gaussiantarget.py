from dataclasses import dataclass, field, InitVar
from typing import NamedTuple, List

import pandas as pd
import numpy as np
from gmpy2 import mpz,bit_mask

@dataclass
class GaussianTargets:
    """
    Describes a single-numeric target variable approximated by a normal distribution, defined by its mean and standard deviation

    Attributes
    ----------
    bit_array : gmpy2.mpz
        A bit_array that covers the whole length of the dataset
    mean : ndarray
        Mean values of the target variables.
    variance : ndarray
        variance of the target variables.

    Parameters
    ----------
    Target : class object that represents a target variable.

    """
    targetvalues : InitVar[pd.DataFrame]
    array_data : np.ndarray = field(init=False)
    bit_array : mpz = field(init=False)
    mean : np.ndarray = field(init = False)
    variance : np.ndarray = field(init = False)
    #TODO: it would be interesting to add a generator instead of a list to do dynamic creation
    def __post_init__(self, targetvalues):
        self.bit_array = bit_mask(targetvalues.shape[0])
        self.array_data = np.asfortranarray(targetvalues.to_numpy(copy=False))
        self.mean = np.mean(targetvalues.values, axis=0)
        self.variance = np.var(targetvalues.values, axis=0)