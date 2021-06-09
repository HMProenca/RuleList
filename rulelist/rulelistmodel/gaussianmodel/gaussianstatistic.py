from dataclasses import dataclass, field, InitVar
from typing import List

import numpy as np
from numba import jit

from rulelist.datastructure.data import Data
from rulelist.rulelistmodel.statistic import Statistic
from rulelist.util.bitset_operations import bitset2indexes


#@jit(nopython=True)
def compute_mean_special(column_data, indices_subgroup):

    sum_values = 0
    for i in range(len(indices_subgroup)):
        sum_values = sum_values + column_data[indices_subgroup[i]]
    return sum_values/len(indices_subgroup)

#@jit(nopython=True)
def compute_mean(values):
    return np.mean(values)

#@jit(nopython=True)
def compute_RSS(values, meanval):
    c = values - meanval
    RSS = np.dot(c, c)
    return RSS

#@jit(nopython=True)
def find2points(values, meandata,bigvalue):
    closest = np.array([bigvalue, bigvalue])
    #closest = values[0:2]
    closest[1] = closest[0]*1.10
    dif = [abs(val - meandata) for val in closest]
    for x in values:
        current_dif = abs(x - meandata)
        if current_dif < dif[0] and x != closest[1]:
            if dif[0] < dif[1] and closest[0] != x:
                closest[1] = closest[0]
                dif[1] = dif[0]
            closest[0] = x
            dif[0] = abs(x - meandata)
        if abs(x - meandata) < dif[1] and x != closest[0]:
            closest[1] = x
            dif[1] = abs(x - meandata)
    return closest, dif

@dataclass(repr=False, eq=False, order=False, unsafe_hash=True, frozen=False)
class GaussianFixedStatistic(Statistic):
    """
    Describes the statistic related to a Gaussian Distribution with fixed mean and variance and corresponding the the
    mean and variance of the dataset.

    Attributes
    ----------
    datastructure : InitVar[Data]
        The dataclass Data that contains all the information regarding the dataset.
    values : InitVar[np.ndarray]
        The values on which to compute the statistics.
    usage : int
        Number of instances covered by the rule.
    mean : List[float]
        Mean of the rule for each target variable.
    variance : List[float]
        Variance of the rule target variable.
    rss : List[float]
        Residual Sum of Squares (RSS) of the rule.
    fixed_parameters: bool = True
        The fact that the parameters of the statistic are fixed.

    Methods
    -------
    _compute_statistics_fixed :
        Computes the statistics of the Gaussian  necessary to compute the encoding. given the fixed values of the mean
        and variance for each target.
    """
    #datastructure : InitVar[Data]
    #bitarray_subgroup : InitVar[list]
    #usage : int = field(default=0,init=False)
    mean : np.ndarray = field(init=False)
    variance : np.ndarray = field(init=False)
    rss :  np.ndarray = field(init=False)
    fixed_parameters : bool = True
    def __post_init__(self, data: Data):
        self.usage, self.number_targers = Statistic.__post_init__(self, data)
        self.mean = data.targets_info.mean
        self.variance = data.targets_info.variance
        self.rss = np.empty(self.number_targers, dtype=np.float64)

    def replace_stats(self, data, bitarray_indices):
        self.usage = self.update_usage(bitarray_indices)
        indices_subgroup = bitset2indexes(bitarray_indices)
        if data.number_targets == 1:
            column_values = data.targets_info.array_data[indices_subgroup,0]
            self.rss[0] = compute_RSS(column_values, self.mean)

            #mean = compute_mean_special(datastructure.target_data_test, indices_subgroup, index_column)
            #self.rss[0]  = compute_RSS_special(datastructure.target_data_test[:,0], indices_subgroup, self.mean)
        elif data.number_targets > 1:
            target_values = data.targets_info.array_data[indices_subgroup,:]
            for icol, column_values in enumerate(target_values.T):
                self.rss[icol] = compute_RSS(column_values, self.mean[icol])
        return self

@dataclass(repr=False, eq=False, order=False, unsafe_hash=True, frozen=False)
class GaussianFreeStatistic(Statistic):
    """
    Describes the statistic related to a Gaussian Distribution with mean and variance unknown, i.e., that they have
    free parameters that have to be averaged when computing their encoding.


    Attributes
    ----------
    datastructure : InitVar[Data]
        The dataclass Data taht contains all the information regarding the dataset.
    values : InitVar[np.ndarray]
        The values on which to compute the statistics.
    usage : int
        Number of instances covered by the rule.
    mean : List[float]
        Mean of the rule for each target variable.
    variance : List[float]
        Variance of the rule target variable.
    rss : List[float]
        Residual Sum of Squares (RSS) of the rule.
    mean_2points : List[float]
        Mean of the 2 points closest to the dataset mean. This value is only computed if fixed_parameters = False, as
        it is necessary for the Bayesian encoding to be valid (see theory in the paper).
    variance_2points : List[float]
        Variance of the 2 points closest to the dataset mean. This value is only computed if fixed_parameters = False, as
        it is necessary for the Bayesian encoding to be valid (see theory in the paper).
    rss_2points : List[float]
        The residual sum of sqsuares of using the mean of the 2 points.
    mean_dataset : List[float]
        Mean of the dataset for each target.
    variance_dataset : List[float]
        Variance of the dataset for each target.
    rss_2dataset : List[float]
        The residual sum of sqsuares of using the mean of the dataset to explain the 2 points.
    fixed_parameters: bool = False
        The fact that the parameters of the statistic are unkown a priori.

    Methods
    -------
    _compute_statistics_free :
        Computes the statistics of the Gaussian  necessary to compute the encoding. given the values assuming that the
         value of the statistics are unkown (not fixed).
    """
    #datastructure : InitVar[Data]
    #bitarray_subgroup : InitVar[list]
    #usage : int = field(init=False)
    mean : np.ndarray = field(init=False)
    variance : np.ndarray = field(init=False)
    rss : List[float] = field(init=False)
    mean_2points : float = field(init=False)
    variance_2points : float = field(init=False)
    rss_2points :  float = field(init=False)
    mean_dataset : float = field(init=False)
    variance_dataset : float = field(init=False)
    rss_2dataset :  float = field(init=False)
    fixed_parameters : bool = False

    def __post_init__(self, data: Data):
        self.usage, self.number_targers = Statistic.__post_init__(self, data)
        self.mean_dataset = data.targets_info.mean
        self.variance_dataset = data.targets_info.variance
        self.mean = np.empty(self.number_targers, dtype=np.float64)
        self.variance = np.empty(self.number_targers, dtype=np.float64)
        self.rss = np.empty(self.number_targers, dtype=np.float64)
        self.mean_2points = np.empty(self.number_targers, dtype=np.float64)
        self.variance_2points = np.empty(self.number_targers, dtype=np.float64)
        self.rss_2points = np.empty(self.number_targers, dtype=np.float64)
        self.rss_2dataset = np.empty(self.number_targers, dtype=np.float64)

    def replace_stats(self,data, bitarray_indices):
        self.usage = self.update_usage(bitarray_indices)
        indices_subgroup = bitset2indexes(bitarray_indices)
        target_values = data.targets_info.array_data[indices_subgroup,:]
        if self.usage > 2:
            for index_column in range(data.number_targets):
                self._compute_statistic_free(data, index_column, target_values[:,index_column])
        elif  self.usage <= 2:
            self._not_enough_points(data)
        return self

    def _compute_statistic_free(self, data, index_column, column_values):
        #column_values = datastructure.target_data_test[indices_subgroup, index_column]
        #column_values = datastructure.targets_info.array_data[indices_subgroup, index_column]
        mean = compute_mean(column_values)
        rss = compute_RSS(column_values, mean)
        self.mean[index_column] = mean
        self.rss[index_column] = rss
        self.variance[index_column] = rss/self.usage
        bigvalue = data.number_instances*data.targets_info.variance[index_column]+data.targets_info.mean[index_column]
        closest2, diff2 = find2points(column_values,data.targets_info.mean[index_column],bigvalue)
        mean2 = compute_mean(closest2)
        self.mean_2points[index_column] = mean2
        self.variance_2points[index_column] = compute_RSS(closest2,mean2)/2
        self.rss_2points[index_column] = diff2[0]**2+diff2[1]**2
        self.rss_2dataset[index_column] = compute_RSS(closest2,self.mean_dataset[index_column])
        return self

    def _not_enough_points(self,data):
        self.mean = np.array([np.nan for it in range(data.number_targets)])
        self.variance = np.array([0 for it in range(data.number_targets)])
        self.rss = np.array([np.inf for it in range(data.number_targets)])
        self.mean_2points = np.array([np.nan for it in range(data.number_targets)])
        self.variance_2points = np.array([np.nan for it in range(data.number_targets)])
        self.rss_2points =np.array([np.nan for it in range(data.number_targets)])
        self.rss_2dataset= np.array([np.nan for it in range(data.number_targets)])
        return self