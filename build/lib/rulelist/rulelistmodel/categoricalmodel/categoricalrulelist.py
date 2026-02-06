from rulelist.rulelistmodel.categoricalmodel.categoricalstatistic import CategoricalFreeStatistic, CategoricalFixedStatistic
from rulelist.rulelistmodel.categoricalmodel.mdl_categorical import length_rule_fixed_categorical, \
    length_rule_free_categorical
from rulelist.rulelistmodel.rulesetmodel import RuleSetModel
from rulelist.util.extra_maths import log2_0

default_rule_statistic_categorical = {
    "discovery": CategoricalFixedStatistic,
    "prediction": CategoricalFreeStatistic
}

compute_default_length_categorical = {
    "discovery": length_rule_fixed_categorical,
    "prediction": length_rule_free_categorical
}


class CategoricalRuleList(RuleSetModel):
    """ Categorical rule list model

    """

    def __init__(self, data, task, max_depth,beam_width,min_support, max_rules, alpha_gain):
        self.max_depth, self.l_combination_pattern, self.l_attribute_item, self.log_prior_class= \
            self._create_constants(data,max_depth)
        super().__init__(data, task, max_depth,beam_width,min_support, max_rules, alpha_gain)
        self.min_support = max(min_support, 4)

    def init_default_statistics(self, data):
        return default_rule_statistic_categorical[self.task](data)

    def init_subgroup_statistics(self, data):
        return CategoricalFreeStatistic(data)

    def compute_default_length(self, default_rule_statistics):
        return compute_default_length_categorical[self.task](self, default_rule_statistics)

    def _create_constants(self, data,max_depth):
        self.max_depth, self.l_combination_pattern, self.l_attribute_item =\
            RuleSetModel._create_constants(self,data,max_depth)
        # compute nml normalizing constant
        #self.log_nml_comp = {(n_points, n_classes): log2(multinomial_with_recurrence(n_classes,n_points))
        #                     if n_points != 0 else 0 for n_points in range(0,datastructure.number_instances+1)
        #                     for n_classes in datastructure.targets_info.number_classes.values()}
        self.log_prior_class = {varname:
                                {category: -log2_0(count/data.number_instances) for category,count in counts.items()}
                                for varname, counts in data.targets_info.counts.items()}
        return self.max_depth, self.l_combination_pattern, self.l_attribute_item, self.log_prior_class

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
                        " usage = " + str(subgroup.statistics.usage)
            n = subgroup.usage
            for varname, usage_per_class in subgroup.statistics.usage_per_class.items():
                text2add += " : target = {}".format(varname)
                text2add += "".join(["Pr({}) = {};".format(category,n_class/n)
                            for category, n_class in usage_per_class.items()])
            text2add += "".join("\n")
        return text2add

    def _add_description_lastrule(self):
        text2add = " ELSE " +\
                   " usage = " + str(self.default_rule_statistics.usage)
        n = self.default_rule_statistics.usage
        if self.task == "discovery":
            for varname, prob_per_class in self.default_rule_statistics.prob_per_classes.items():
                text2add += " : target = {}".format(varname)
                text2add += "".join(["Pr({}) = {};".format(category, prob)
                                 for category, prob in prob_per_class.items()])
        else:
            for varname, usage_per_class in self.default_rule_statistics.usage_per_class.items():
                text2add += " : target = {}".format(varname)
                text2add += "".join(["Pr({}) = {};".format(category, n_class / n)
                            for category, n_class in usage_per_class.items()])
        return text2add