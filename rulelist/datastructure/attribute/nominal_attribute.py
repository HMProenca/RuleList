from dataclasses import field, dataclass
from functools import partial
from typing import List, Tuple, Any, AnyStr, Dict

import numpy as np
import pandas as pd

from rulelist.datastructure.attribute.attribute import Item, Attribute
from rulelist.util.bitset_operations import indexes2bitset


def activation_nominal(df: pd.DataFrame, attribute_name: AnyStr, category: Any) -> pd.DataFrame:
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
    activated_indexes = df[attribute_name] == category
    return activated_indexes

@dataclass
class NominalAttribute(Attribute):  # TODO: add sets of categories with OR logic (for now Nominal is equal BInary)
    """
    Describes a nominal attribute or variable. Inherits from class Attribute.

    Attributes
    ----------
    categories : np.ndarray
        Array of categories.
    items : List[Item]
        List of items that are made from the values covered by the categories of this attribute.

    Parameters
    ----------
    Attribute : class object that represents a variable.

    Methods
    -------
    create_items
        Creates the items from the categories of the nominal attribute with one operator. Example: x == blue_eyes

    """
    categories : np.ndarray = field(default_factory=list, init=False)
    cardinality_operator : Dict[int,int] =field(init=False)
    def __post_init__(self):
        self.categories = self.values.unique()
        self.items, self.cardinality_operator =  self.create_items()

    #TODO: expand make items simple nominal to sets of items with the logical OR
    def create_items(self) -> Tuple[List[Item], Dict[int, int]]:
        """ Creates a list of items from the nominal atrribute.

        Makes a list of items using equality relationship with the categories. Example: x= blue_eyes could be the
        description of one of the items, for the NominalAttribute.name = "eye_colour".

        Returns
        ----------
        List[Item] : List of Items
            A list of all items based on the possible categories (only with equality relationships, not logical ORs).
        """
        self.cardinality_operator = {1: len(self.categories)}
        number_operators =  1
        for category in self.categories:
            vector_category = np.where(self.values == category)[0]
            bit_array = indexes2bitset(vector_category)
            description = str(self.name) + " = " + str(category)
            activation_function = partial(activation_nominal, attribute_name=self.name, category=category)
            self.items.append(Item(bit_array,self.name, description, number_operators,activation_function))
        return self.items, self.cardinality_operator


