from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple, AnyStr, Iterator, Dict

import numpy as np
from pandas import DataFrame

from rulelist.datastructure.attribute.attribute import Attribute, Item
from rulelist.util.bitset_operations import indexes2bitset


def activation_numeric(df: DataFrame, attribute_name: AnyStr, minval: float, maxval: float) -> DataFrame:
    """Checks in which instances the numerical conditions are True.

    Parameters
    ----------
    df : pandas.DataFrame
        List of items that describe single-numeric attribute.
    attribute_name : str
        Name of attribute.
    minval: float
        Minimum value in the condition x >= minval.
    maxval
        Maximum value in the condition x < maxval.

    Returns
    ----------
    activated_indexes : np.ndarray
        Boolean array with True for values where the conditions are true.
    """
    activated_indexes = (df[attribute_name] >= minval) & (df[attribute_name] < maxval)
    return activated_indexes


def find_cutpoints(values: np.ndarray, n_cutpoints: int) -> Tuple[np.ndarray, int]:
    """ Finds the n quantile values as if done with equal frequency binning.

    Parameters
    ----------
    values : np.ndarray
        Array of values to discretize.
    n_cutpoints : int
        Number of cut points selected.

    Returns
    ----------
    value_quantiles : np.ndarray
        Array of the quantile values.
    real_ncutpoints : int
        In case the values do not allow n_cutpoints it returns a smaller value.
    """
    if n_cutpoints > len(values):
        n_cutpoints =  len(values)
    quantile_percentage = [1 / (n_cutpoints + 1) * ncut for ncut in range(0, n_cutpoints + 2)]
    value_quantiles = np.nanquantile(values, quantile_percentage, interpolation='midpoint')[1:-1]
    # if np.isnan(val_quantiles).any(): continu
    value_quantiles = np.unique(value_quantiles)
    real_ncutpoints = len(value_quantiles)
    return value_quantiles, real_ncutpoints

def create_item(indexes, variable_name, min_val, max_val, description, number_operations):
    """ Creates a class of type Item from the values of a NumericAttribute.

    Parameters
    ----------
    indexes : np.ndarray
        Array of indexes where the item is present in the training datastructure.
    variable_name : str
        Name of the attribute/variable that this item is attached to.
    min_val : float
        Minimum value covered by this item. item > min_val.
    max_val : float
        Maximum value covered by this item. item < max_val.
    description : str
        Text describing the interval defined by the item. item < max_val = 1;  min_val <  item < max_val = 2.
    number_operations : int
        Number of logical operators used to define the interval.
    Returns
    ----------
    Item : Item class object
        Item with the characteristics described by the arguments.
    """
    bit_array = indexes2bitset(indexes)
    activation_function = partial(activation_numeric, attribute_name=variable_name, minval=min_val, maxval=max_val)
    return Item(bit_array, variable_name, description, number_operations, activation_function)

@dataclass
class NumericAttribute(Attribute):
    """
    Describes a single-numeric attribute or variable. Inherits from class Attribute.

    Attributes
    ----------
    items : List[Item]
        List of items that describe single-numeric attribute.
    n_items : int
        Number of items in this attribute.

    Parameters
    ----------
    Attribute : class object that represents a variable.

    Methods
    -------
    create_items_numeric
        Creates the items by making binary partitions of the values using the cutpoints of equal frequency binning.
    """
    n_cutpoints : int
    discretization : AnyStr
    items : List[Item] = field(default_factory=list, init=False)
    cardinality_operator: Dict[int,int] = field(default_factory=dict, init=False)
    #TODO: it would be interesting to add a generator instead of a list to do dynamic creation
    def __post_init__(self):
        self.items, self.cardinality_operator = self.create_items()

    def create_items(self) -> Tuple[List[Item], Dict[int, int]]:
        """ Creates a list of items from the numerical atrribute.

        Makes a list of items using equal frequency binning, ignoring NANs, based on the values of the Numeric attribute

        Returns
        ----------
        List[Item] : List of Items
            A list of all items based on the possible combinations of cutpoints.
        """
        value_quantiles, self.n_cutpoints = find_cutpoints(self.values, self.n_cutpoints)
        self.cardinality_operator = {1:0,2:0}
        for iq, value_quantile1 in enumerate(value_quantiles):  # makes binary intervals x<val and x >=val
            # condition x<val
            index_down = np.where(self.values < value_quantile1)[0]
            description_down = self.name + " < " + str(value_quantile1)
            self.items.append(create_item(index_down,variable_name= self.name, min_val=np.NINF, max_val=value_quantile1,
                                          description = description_down,number_operations=1))
            self.cardinality_operator[1] += 1
            if self.n_cutpoints == 1: break # if there is only one unique value we only need one item
            # condition x >=val
            index_up = np.where(self.values >= value_quantile1)[0]
            description_up = self.name + " >= " + str(value_quantile1)
            self.items.append(create_item(index_up,variable_name= self.name, min_val=value_quantile1, max_val=np.inf,
                                          description = description_up,number_operations=1))
            self.cardinality_operator[1] += 1
            # conditions val1 <= x < val2
            for value_quantile2 in value_quantiles[iq + 1:]:
                index_interval = np.where((self.values >= value_quantile1) & (self.values < value_quantile2))[0]
                description_interval = str(value_quantile1) + " <= " + self.name + " < " + str(value_quantile2)
                self.items.append(create_item(index_interval,variable_name= self.name, min_val=value_quantile1,
                                              max_val=value_quantile2,description = description_interval,
                                              number_operations=2))
                self.cardinality_operator[2] += 1
        return self.items,self.cardinality_operator

    def generate_items(self,candidate) -> Iterator[Item]:
        #TODO: make dynamic generation of items based on "candidate"
        for item in self.items:
            yield item