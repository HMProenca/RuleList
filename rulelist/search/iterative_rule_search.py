# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:52:14 2019

@author: gathu
"""
from rulelist.datastructure.data import Data
from rulelist.rulelistmodel.categoricalmodel.categoricalrulelist import CategoricalRuleList
from rulelist.rulelistmodel.gaussianmodel.gaussianrulelist import GaussianRuleList
from rulelist.search.beam.itemset_beamsearch import find_best_rule

rule_list_models = {
    "gaussian":GaussianRuleList,
    "categorical":CategoricalRuleList
}

def _fit_rulelist(X, Y, target_model, max_depth, beam_width, min_support,
                  n_cutpoints, task, discretization, max_rules, alpha_gain):
    """ this function finds the rule/subgroup list given the selected datastructure,
    target, target type using the Minimum Description Length (MDL) principle 
    formulation.
    """
    # initialize the datastructure in the needed format for the analysis
    data = Data(input_data=X,n_cutpoints=n_cutpoints,discretization=discretization,
                target_data=Y,target_model=target_model, min_support = min_support)
    rulelist = rule_list_models[data.target_model](data, task, max_depth, beam_width,min_support, max_rules, alpha_gain)
    rulelist = greedy_and_beamsearch(data,rulelist) #TODO: this should be a method of rulelist
    rulelist.add_description()
    return rulelist

def greedy_and_beamsearch(data,rulelist):
    while True:
        print("Iteration: " + str(rulelist.number_rules+1))
        subgroup2add = find_best_rule(rulelist, data)
        if subgroup2add.score <= 0: break
        rulelist = rulelist.add_rule(subgroup2add,data)
        if rulelist.number_rules >= rulelist.max_rules: break
    return rulelist