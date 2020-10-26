from dataclasses import InitVar, dataclass, field

from gmpy2 import popcount

from rulelist.datastructure.data import Data


@dataclass
class Statistic:
    """
    Describes the skeleton of a statistic object

    Attributes
    ----------
    datastructure : InitVar[Data]
        The dataclass Data taht contains all the information regarding the dataset.
    values : InitVar[np.ndarray]
        The values on which to compute the statistics.
    usage : int
        Number of instances covered by the rule.

    """
    data : InitVar[Data]
    usage : int = field(default= 0, init=False)
    number_targets : int = field(init=False)
    def __post_init__(self, data: Data):
        self.number_targets = data.number_targets
        return self.usage, self.number_targets

    def update_usage(self, bitarray_indices):
        self.usage = popcount(bitarray_indices)
        return self.usage

    def replace_stats(self,data, bitarray_indices):
        self.usage = self.update_usage(bitarray_indices)
        return self
