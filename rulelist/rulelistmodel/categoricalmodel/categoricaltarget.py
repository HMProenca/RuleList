
from dataclasses import dataclass, field, InitVar
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from gmpy2 import mpz, bit_mask

from rulelist.util.bitset_operations import indexes2bitset


@dataclass
class CategoricalTarget:
    """
    Describes a nominal target variable approximated by a single-nominal distribution, defined by the counts per category.

    Attributes
    ----------
    categories : List[Any]
        List of the categories of the nominal variable.
    bit_array : Dict[gmpy2.mpz]
        A dictionary of bit_arrays, one for each category
    counts : Dict[int]
        Number of counts per category.

    Parameters
    ----------
    Target : a generic class object that represents a target variable.

    """
    target_values : InitVar[pd.DataFrame]
    categories : Dict[Any, np.ndarray]  = field(init=False)
    number_classes: Dict[Any, int] = field(init=False)
    bit_array: mpz = field(default=mpz(0), init=False)
    bit_arrays_var_class: Dict[Any, Dict[Any, mpz]] = field(default_factory=dict, init=False)
    counts: Dict[Any, np.ndarray] = field(default_factory=dict,init=False)
    prob_var_class : Dict[Any, Dict[Any, float]] = field(default_factory=dict,init=False)
    def __post_init__(self, target_values):
        self.bit_array = bit_mask(target_values.shape[0])
        self.categories = {colname: colvals.unique() for colname, colvals in target_values.iteritems()} #ignores NANs values
        self.number_classes = {colname: len(array_uniques) for colname, array_uniques in self.categories.items()}
        if any([nunique == 1 for nunique in self.number_classes.values()]):
            raise ValueError("There is at least one target variable with only one class label. Please only add targets with 2 or more class labels.")
        self.bit_arrays_var_class, self.counts,  self.prob_var_class = self.init_bitarrays_class(target_values)

    def init_bitarrays_class(self, target_values) -> Tuple[Dict[Any, np.ndarray],Dict[Any, np.ndarray]] :
        """ Initializes the bit array values for each category.

        Returns
        ----------
        Dict[gmpy2.mpz] :
            A dictionary of the bitarray values.
        """
        for namecol, colvals in target_values.iteritems():
            self.bit_arrays_var_class[namecol] = dict()
            self.counts[namecol] = dict()
            self.prob_var_class[namecol] = dict()
            for icat, category in enumerate(self.categories[namecol]):
                category_indexes = np.where(colvals.values == category)[0]
                self.bit_arrays_var_class[namecol][category] = indexes2bitset(category_indexes)
                self.counts[namecol][category] = len(category_indexes)
                self.prob_var_class[namecol][category] = self.counts[namecol][category]/target_values.shape[0]
        return self.bit_arrays_var_class, self.counts, self.prob_var_class

