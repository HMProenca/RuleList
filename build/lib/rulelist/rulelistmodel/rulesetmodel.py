from copy import deepcopy

from gmpy2 import mpz, bit_mask, popcount

from rulelist.mdl.mdl_base_codes import uniform_combination_code, uniform_permutation_code
from rulelist.rulelistmodel.data_encoding import compute_length_data
from rulelist.rulelistmodel.model_encoding import compute_item_length, compute_length_model
from rulelist.rulelistmodel.statistic import Statistic


class RuleSetModel():
    """ rule set model

    """

    def __init__(self, data, task, max_depth, beam_width,min_support, max_rules, alpha_gain):
        self.task = task
        self.target_model = data.target_model
        self.alpha_gain = alpha_gain
        self.number_rules = 0
        self.targets_info = data.targets_info
        #TODO: substitute width and depth and max rules for a search query
        self.beam_width = beam_width
        self.min_support = data.number_instances*min_support if min_support < 1.0 else min_support

        self.max_depth = max_depth if max_depth < data.number_attributes else data.number_attributes
        self.max_rules = max_rules

        # rule set characteristics
        self.bitset_covered = mpz()
        self.support_covered = 0
        self.bitset_uncovered = bit_mask(data.number_instances)
        self.support_uncovered = data.number_instances

        # subgroups
        # The bitset and subgroup.bit_array of the subgroups is as if they were an independent pattern and not in the ordered rule list
        self.subgroups = []
        self.bitset_rules = []
        # string format of the rule set
        self.description = "There are no rules to show."
        # regarding BEAM search
        self.tmp_subgroup_statistic = self.init_subgroup_statistics(data)
        self.tmp_default_statistic  = self.init_default_statistics(data)
        # MDL characteristics of the model
        self.length_model = 0
        self.default_rule_statistics = self.init_default_statistics(data)
        self.default_rule_statistics = self.default_rule_statistics.replace_stats(data, self.bitset_uncovered)

        self.length_data = self.compute_default_length(self.default_rule_statistics)
        self.length_original = self.length_data
        self.length_defaultrule = self.length_data # when there are no more rules it is the same
        self.length_ratio = 1.0

    def add_rule(self, subgroup2add, data):
        self.number_rules += 1
        self._add_subgroup2list(subgroup2add)
        self.length_model = compute_length_model(self)
        self.default_rule_statistics = self.default_rule_statistics.replace_stats(data, self.bitset_uncovered)
        self.length_defaultrule = self.compute_default_length(self.default_rule_statistics)
        self.length_data = compute_length_data(self) # self.length_defaultrule needs to be computed before!
        self._compute_length_ratio()
        return self

    def _create_constants(self, data,max_depth):
        self.max_depth = max_depth if max_depth < data.number_attributes else data.number_attributes
        if data.discretization == 'static':
            self.l_variables_in_pattern = {size : uniform_combination_code(size, data.number_attributes)
                                           for size in range(1, self.max_depth + 1)}
        elif data.discretization == 'dynamic':
            self.l_variables_in_pattern = {size : uniform_permutation_code(size, data.number_attributes)
                                           for size in range(1, self.max_depth + 1)}
        self.l_attribute_item = {(attribute_name, n_operators) : l_item for attribute in data.attributes
                                 for attribute_name, n_operators, l_item in compute_item_length(attribute)}
        return self.max_depth, self.l_variables_in_pattern, self.l_attribute_item

    def _compute_length_ratio(self):
        """ In case the variance is small the length becomes negative. This is merely an artifact of scale.
        """
        if self.length_original > 0:
            self.length_ratio = (self.length_data + self.length_model) / self.length_original
        elif self.length_original < 0:
            self.length_ratio = self.length_original / (self.length_data + self.length_model)
        return self

    def _add_subgroup2list(self, subgroup2add):
        self.bitset_covered = self.bitset_covered | subgroup2add.bitarray
        self.support_covered = popcount(self.bitset_covered)
        self.bitset_uncovered = self.bitset_uncovered & ~ subgroup2add.bitarray
        self.support_uncovered = popcount(self.bitset_uncovered)
        self.bitset_rules.append(subgroup2add.bitarray)
        self.subgroups.append(deepcopy(subgroup2add))
        return self


    def init_subgroup_statistics(self,data):
        return Statistic(data)

    def init_default_statistics(self, data):
        return Statistic(data)

    def compute_default_length(self,default_statistics):
        return None

    def add_description_antecedent(self, newsubgroup, attributes):
        pass
    def add_description_consequent(self, newsubgroup, attributes):
        pass
    def create_constants(self, data):
        pass