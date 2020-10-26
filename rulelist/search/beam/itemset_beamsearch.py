# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:09:11 2019

@author: gathu
"""
from functools import reduce

import numpy as np
from gmpy2 import popcount, bit_mask

from rulelist.datastructure.subgroup import Subgroup
from rulelist.rulelistmodel.gain_add_rule import compute_delta_score, compute_statistics_newrules
from rulelist.search.beam.beam import Beam


def refine_subgroup(rulelist,data,candidate2refine,beam,subgroup2add):
    """ Expands a subgroup by adding an item from all other variables not included in the subgroup.

    """
    bitarray_candidate = reduce((lambda x, y: x & y), (item.bitarray for item in candidate2refine)) \
        if candidate2refine != [] else bit_mask(data.number_instances)
    #TODO: move this computation depending if it is a rule list or a rule set
    bitarray_candidate = bitarray_candidate & rulelist.bitset_uncovered
    variable_list = [item.parent_variable for item in candidate2refine]
    for attribute in filter(lambda x: x.name not in variable_list, data.attributes):
        #for item in attribute.generate_items(candidate2refine):
        for item in attribute.items:
            bitarray_newcandidate = bitarray_candidate & item.bitarray
            usage = popcount(bitarray_newcandidate)
            if usage >= rulelist.min_support:
                new_subgroup_statistics, new_default_rule_statistics = \
                    compute_statistics_newrules(rulelist, data,bitarray_newcandidate)
                new_candidate = candidate2refine + [item]
                score, gain_data, gain_model = compute_delta_score(rulelist, new_candidate, new_subgroup_statistics, new_default_rule_statistics)
            else:
                score = np.NINF
            if score > subgroup2add.score:
                subgroup2add.update(new_candidate, new_subgroup_statistics, gain_data, gain_model, score)
            if score > beam.min_score and set([item.description for item in new_candidate]) not in beam.set_patterns:
                beam.replace(new_candidate, score, usage)
                #print("Subgroup: {} ; score : {}".format([pat.parent_variable for pat in new_candidate],score))
    return beam, subgroup2add

def find_best_rule(rulelist, data):
    """ Finds the best rule using beam search given the rule list so far and the datastructure.
    """
    subgroup2add = Subgroup()
    beam = Beam(rulelist.beam_width)
    for depth in range(rulelist.max_depth):
        candidates = [pattern for ip, pattern in enumerate(beam.patterns)
                      if pattern not in beam.patterns[:ip]
                      and len(pattern) == depth
                      and beam.array_support[ip] > rulelist.min_support]
        beam = beam.clean()
        for candidate2refine in candidates:
            beam, subgroup2add = refine_subgroup(rulelist,data,candidate2refine,beam,subgroup2add)
    print("Gain datastructure: {} ; gain model : {} ; gain: {}".format(subgroup2add.delta_data,subgroup2add.delta_model,subgroup2add.score))
    return subgroup2add
