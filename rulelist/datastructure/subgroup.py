from copy import deepcopy
from functools import reduce
from typing import List

import numpy as np
from gmpy2 import mpz, popcount

from rulelist.datastructure.attribute.attribute import Item


class Subgroup():
    """
    Describes a Subgroup, which is a Logical combination of items.

    Attributes
    ----------
    pattern : List[Item]
        A list of items that compose the pattern.
    statistic : List[Any]
        List of the statistics for each target value.
    delta_data : float
        The local improvement in the datastructure encoding of adding this subgroup to the rule list.
    delta_model : float
        The local improvement (always negative) in the model encoding of adding this subgroup to the rule list.
    score : float
        A weighted composition of (delta_data + delta_model). If divided by the usage it equals the normalized alpha_gain,
        and if not it equals the absolute alpha_gain.
    usage : int
        Number of instances covered by the description of the subgroup given its position in the rule list.
    support : List[Item]
        Number of instances covered by the description of subgroup, treating the subgroup as independent from the rule
        list.
    bitarray : mpz
        Bit array of covered instances. popcount(bitarray) = support

    Methods
    -------
    _compute_bitarray
        Computes the bitarray and support from the list of items.

    """
    def __init__(self):
        self.pattern = []
        self.statistics = None
        self.delta_data = np.NINF
        self.delta_model = np.NINF
        self.score = np.NINF
        self.usage = 0
        self.variable_list = []
        self.support = 0
        self.bitarray = mpz()
        self.size = 0

    def update(self,new_candidate,new_subgroup_statistics,gain_data, gain_model, score):
        self.pattern = new_candidate
        self.statistics = deepcopy(new_subgroup_statistics)
        self.usage = self.statistics.usage
        self.delta_data = gain_data
        self.delta_model = gain_model
        self.score = score
        # Note that the bitarray only consider the pattern alone, not in the ordered rule list
        self.bitarray, self.support = self._compute_bitarray()
        self.variable_list = {item.parent_variable for item in self.pattern}
        self.size = len(self.pattern)
        return self

    def _compute_bitarray(self):
        self.bitarray = reduce(lambda x, y: x & y, [item.bitarray for item in self.pattern])
        self.support = popcount(self.bitarray)
        return self.bitarray, self.support