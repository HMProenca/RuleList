from dataclasses import dataclass, field
from typing import List, Literal

import pandas as pd
from pandas.api.types import is_numeric_dtype

from rulelist.datastructure.attribute.attribute import Attribute
from rulelist.datastructure.attribute.nominal_attribute import NominalAttribute
from rulelist.datastructure.attribute.numeric_attribute import NumericAttribute
from rulelist.rulelistmodel.categoricalmodel.categoricaltarget import CategoricalTarget
from rulelist.rulelistmodel.gaussianmodel.gaussiantarget import GaussianTargets

#TODO: add location and multivariate gaussian
init_target = {
    "gaussian" : GaussianTargets,
    "categorical" : CategoricalTarget
};

@dataclass
class Data:
    """
    Contains all information regarding the descriptive variables of the dataset.

    It will be composed of a list of attributes, each relating to a variable and its characteristics.

    Attributes
    ----------
    datastructure : pd.DataFrame
        It contains a view to the original input dataset.
    attributes : List[Attribute]
        A list of the variables and its characteristics.

    Methods
    -------
    init_attributes
        Initializes all the attributes with their respective values.
    """
    input_data : pd.DataFrame
    n_cutpoints : int
    discretization: Literal["static", "sequential", "dynamic"]
    target_data: pd.DataFrame
    target_model: Literal["gaussian", "single-nominal"]
    min_support: int
    attributes: List[Attribute] = field(default_factory=list, init=False)
    number_attributes: int = field(init=False)
    attribute_names: set = field(init=False)
    target_names: set = field(init=False)
    targets_info: classmethod = field(init=False)
    number_targets: int = field(init=False)
    number_instances: int = field(init=False)
    def __post_init__(self):
        self.input_data = pd.DataFrame(self.input_data) #in case it is a series it will be transformed to DataFrame
        self.target_data = pd.DataFrame(self.target_data)
        if self.input_data.shape[0] != self.target_data.shape[0]:
            raise IndexError('Input datastructure and Target datastructure should have the same number of instances')
        self.attribute_names = set(self.input_data.columns)
        self.target_names = set(self.target_data.columns)
        self.attributes = self._init_attributes()
        self.number_attributes = len(self.attributes)
        self.targets_info = init_target[self.target_model](self.target_data)
        self.number_targets = self.target_data.shape[1]
        self.number_instances = len(self.input_data.index)

    def _init_attributes(self) -> List[Attribute]:
        """ Initializes all attributes based on their values and type.

        It uses pandas.api.types function is_numeric_dtype to identify if a variable is single-numeric or not.
        This means that nominal variables cannot be integer values, as it happens usually. Another possibility to deal
        with nominal variables that are integers is to convert them directly in the DataFrame to single-nominal or
        to object dtype.

        Returns
        ----------
        attributes : List[Attribute]
            It returns a list of attributes already initialized.
        """
        #self.attributes = list() # clean in case it has previous values
        #TODO: stop hardcoding max_operators and ask to the user, special for nominal!
        for name, values in self.input_data.iteritems():
            if is_numeric_dtype(self.input_data[name]):
                max_operators = 2
                self.attributes.append(NumericAttribute(name, values, max_operators,self.min_support, self.n_cutpoints, self.discretization))
            else:  # Nominal or Binary
                max_operators = 1
                self.attributes.append(NominalAttribute(name, values, max_operators,self.min_support))
        return self.attributes