from functools import reduce
from math import log, sqrt

from gmpy2 import popcount, xmpz
from scipy.special.cython_special import gammaln
import numpy as np

from mdlrulelist.rulelistmodel.gaussianmodel.mdl_gaussian import length_rule_fixed_gaussian, length_rule_free_gaussian
from mdlrulelist.rulelistmodel.gaussianmodel.gaussianstatistic import GaussianFixedStatistic, GaussianFreeStatistic
from mdlrulelist.rulelistmodel.rulesetmodel import RuleSetModel

default_rule_statistic_gaussian = {
    "discovery": GaussianFixedStatistic,
    "prediction": GaussianFreeStatistic
}

compute_default_length_gaussian = {
    "discovery": length_rule_fixed_gaussian,
    "prediction": length_rule_free_gaussian
}


class GaussianRuleList(RuleSetModel):
    """ rule set model

    """

    def __init__(self, data, task, max_depth,beam_width,min_support, max_rules, alpha_gain):
        self.max_depth, self.l_combination_pattern, self.l_attribute_item = self._create_constants(data, max_depth)
        super().__init__(data, task, max_depth,beam_width,min_support, max_rules, alpha_gain)
        self.min_support = max(min_support,4)

    def init_default_statistics(self, data):
        return default_rule_statistic_gaussian[self.task](data)

    def init_subgroup_statistics(self, data):
        return GaussianFreeStatistic(data)

    def compute_default_length(self, default_rule_statistics):
        return compute_default_length_gaussian[self.task](self, default_rule_statistics)


    def _create_constants(self, data,max_depth):
        self.max_depth, self.l_combination_pattern, self.l_attribute_item =\
            RuleSetModel._create_constants(self,data,max_depth)
        return self.max_depth, self.l_combination_pattern, self.l_attribute_item

    def add_description(self):
        self.description = self._add_description_rules() + self._add_description_lastrule()
        return self

    def _add_description_rules(self):
        text2add = ""
        for isub, subgroup in enumerate(self.subgroups):
            text2add += "If" if isub == 0 else "ELSE IF"
            for iit, item in enumerate(subgroup.pattern):
                text2add += " " + item.description + " "
                text2add += " AND " if iit < len(subgroup.pattern)-1 else ""
            text2add += " THEN " + \
                        " usage = " + str(subgroup.statistics.usage) + \
                        "; mean = " + str(subgroup.statistics.mean) + \
                        "; std = " + str(np.sqrt(subgroup.statistics.variance))+ \
                        " \n"
        return text2add

    def _add_description_lastrule(self):
        text2add = " ELSE " +\
                   " usage = " + str(self.default_rule_statistics.usage)+ \
                   "; mean = " + str(self.default_rule_statistics.mean) + \
                   "; std = " + str(np.sqrt(self.default_rule_statistics.variance))
        return text2add