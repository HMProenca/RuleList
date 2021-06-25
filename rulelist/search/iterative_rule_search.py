# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:52:14 2019

@author: Hugo Proenca
"""
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