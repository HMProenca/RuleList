# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:52:14 2019

@author: Hugo Proenca
"""
from rulelist.datastructure.data import Data
from rulelist.rulelistmodel.categoricalmodel.categoricalrulelist import CategoricalRuleList
from rulelist.rulelistmodel.gaussianmodel.gaussianrulelist import GaussianRuleList
from rulelist.search.beam.itemset_beamsearch import find_best_rule


def greedy_and_beamsearch(data,rulelist):
    while True:
        print("Iteration: " + str(rulelist.number_rules+1))
        subgroup2add = find_best_rule(rulelist, data)
        #print('Variance : {} ; delta_data: {} ; support ; {}'.format(subgroup2add.statistics.variance ,subgroup2add.delta_data,subgroup2add.usage ))
        if subgroup2add.score <= 0: break
        rulelist = rulelist.add_rule(subgroup2add,data)
        #if rulelist.number_rules >= rulelist.max_rules: break
    return rulelist


def _fit_rulelist(input_data, target_data, target_model, max_depth, beam_width, iterative_beam_width,
                  n_cutpoints, task, discretization, max_rules, alpha_gain, min_support=1):
    """
    Fit a rule list using the same parameters as the legacy iterative search routine.

    Parameters mirror the original public API and are kept for backward compatibility; the
    iterative_beam_width argument is accepted but not used.
    """
    data = Data(input_data=input_data, n_cutpoints=n_cutpoints, discretization=discretization,
                target_data=target_data, target_model=target_model, min_support=min_support)

    if target_model == "categorical":
        rulelist = CategoricalRuleList(data, task, max_depth, beam_width, min_support, max_rules, alpha_gain)
    else:
        rulelist = GaussianRuleList(data, task, max_depth, beam_width, min_support, max_rules, alpha_gain)

    rulelist = greedy_and_beamsearch(data, rulelist)
    rulelist.add_description()
    return rulelist
