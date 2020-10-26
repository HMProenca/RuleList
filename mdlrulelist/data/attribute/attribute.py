from dataclasses import dataclass, field
from typing import List, AnyStr, Iterator, Tuple, Any, Dict

import numpy as np
import pandas as pd
from gmpy2 import mpz


@dataclass
class Item:
    """ Describes an item of an attribute.

    The item is the lowest level representation of the attribute. It is defined by at least one condition and,
     for example, in the case of a NominalAttribute it can be given by the condition: x = blue_eyes; and in the
     NumericAttribute by: x < 3;

    Attributes
    ----------
    bit_array : gmpy2.mpz
        Bit representation of the indexes covered by the item's condition.
    description : str
        Text describing the item.
    numper_operators : int
        Number of operators necessary to describe the item.
    activation_function : object
        Partial function applied to DataFrame that returns boolean vector of instances where item is "present".
    """
    bitarray: mpz
    parent_variable: AnyStr
    description: AnyStr
    number_operators: int
    activation_function: object

@dataclass
class Attribute:
    """ Describes an explainable variable.

    Contains all information regarding a certain attribute. This is the parent class for NumericAttribute and
    NominalAttribute, which add extra specific information to this.

    Attributes
    ----------
    name : str
        Name of the attribute.
    values : np.ndarray
        Vector of values associated with attribute.
    max_operators : int
        Maximum number of operators allowed for this variable.
    min_support : int
        Minimum support of numer of instances covered by an item or pattern
    """
    name: AnyStr
    values: pd.Series
    max_operators: int
    min_support: Any # it can be an int or a float
    cardinality_operator : Dict[int, int] =field(init=False)
    items : List[Item] = field(default_factory=list, init=False)

    def generate_items(self,candidate) -> Iterator[Item]:
        for item in self.items:
            yield item