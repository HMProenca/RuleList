from functools import reduce
from typing import Dict
from operator import add

from mdlrulelist.mdl.mdl_base_codes import log_multinomial
from mdlrulelist.rulelistmodel.categoricalmodel.categoricalstatistic import CategoricalFreeStatistic,CategoricalFixedStatistic
from mdlrulelist.util.build.extra_maths import log2_0


def categorical_free_encoding(statistics, varname):
    codelength = statistics.usage*log2_0(statistics.usage)
    codelength -= sum([n_class*(log2_0(n_class)) for n_class in statistics.usage_per_class[varname].values()])
    codelength += log_multinomial(statistics.number_classes[varname],statistics.usage)
    return codelength

def categorical_fixed_encoding(rulelist, statistics, varname):
    codelength = sum([n_class*(rulelist.log_prior_class[varname][category])
                      for category, n_class in statistics.usage_per_class[varname].items()])
    return codelength

def length_rule_free_categorical(rulelist : classmethod, statistics : CategoricalFreeStatistic):
    l_free = sum([categorical_free_encoding(statistics, varname)
                   for varname in statistics.usage_per_class.keys()])
    return l_free

def length_rule_fixed_categorical(rulelist : classmethod, statistics : CategoricalFixedStatistic):
    l_fixed = sum([categorical_fixed_encoding(rulelist, statistics, varname)
                   for varname, counts_per_class in statistics.usage_per_class.items()])
    return l_fixed