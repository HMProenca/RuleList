from collections import OrderedDict,defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any

from gmpy2 import popcount

from mdlrulelist.data.data import Data
from mdlrulelist.rulelistmodel.statistic import Statistic

@dataclass(repr=True, eq=False, order=False, unsafe_hash=True, frozen=False)
class CategoricalFixedStatistic(Statistic):
    usage_per_class : Dict[Any, Dict[Any, int]] = field(init=False)
    number_classes : Dict[Any, int] = field(init=False)
    prob_per_classes : Dict[Any, Dict[Any, float]] = field(init=False)
    def __post_init__(self, data: Data):
        self.usage, self.number_targets = Statistic.__post_init__(self, data)
        self.number_classes = data.targets_info.number_classes
        self.usage_per_class = {varname: dict() for varname in data.target_names}
        self.prob_per_classes = data.targets_info.prob_var_class

    def replace_stats(self,data,indices_bitarray):
        self.update_usage(indices_bitarray)
        for varname, bit_arrays_class in data.targets_info.bit_arrays_var_class.items():
            for category in data.targets_info.categories[varname]:
                self.usage_per_class[varname][category] = popcount(indices_bitarray & bit_arrays_class[category])
        return self

@dataclass(repr=True, eq=False, order=False, unsafe_hash=True, frozen=False)
class CategoricalFreeStatistic(Statistic):
    usage_per_class : Dict[Any, Dict[Any, int]] = field(init=False)
    number_classes : Dict[Any, int] = field(init=False)
    def __post_init__(self, data: Data):
        self.usage, self.number_targets = Statistic.__post_init__(self, data)
        self.number_classes = data.targets_info.number_classes
        self.usage_per_class = {varname: dict() for varname in data.target_names}

    def replace_stats(self,data, index_bitarray):
        self.usage = self.update_usage(index_bitarray)
        for varname, bit_arrays_class in data.targets_info.bit_arrays_var_class.items():
            for category in data.targets_info.categories[varname]:
                self.usage_per_class[varname][category] = popcount(index_bitarray & bit_arrays_class[category])
        return self
