from rulelist.mdl.mdl_base_codes import universal_code_integers
from rulelist.rulelistmodel.categoricalmodel.mdl_categorical import length_rule_free_categorical
from rulelist.rulelistmodel.gaussianmodel.mdl_gaussian import length_rule_free_gaussian

length_rule_free = {
    "gaussian": length_rule_free_gaussian,
    "categorical": length_rule_free_categorical
}

def compute_delta_data(rulelist, new_subgroup_statistics, new_default_rule_statistics):
    """ Computes the alpha_gain (delta) in code length of adding one rule two the model.

    It needs 3 components:

    """
    l_subgroup =  length_rule_free[rulelist.target_model](rulelist, new_subgroup_statistics)
    l_newdefault = rulelist.compute_default_length(new_default_rule_statistics)
    gain = rulelist.length_defaultrule - l_newdefault - l_subgroup
    return gain

def compute_delta_model(rulelist, new_candidate):
    """ Computes the alpha_gain (delta) of adding a new candidate to the rule list.
    Notice that a positive alpha_gain means that the overall length of the rule list diminishes by adding the new candidate.
    Model Gain is always negative as adding something to the rule list can only increase the model complexity.
    """
    #delta_rules = rulelist.l_universal[rulelist.number_rules] - rulelist.l_universal[rulelist.number_rules+1]
    #l_pattern_length = rulelist.l_universal[len(new_candidate)]
    delta_rules = universal_code_integers(rulelist.number_rules) - universal_code_integers(rulelist.number_rules + 1)
    l_pattern_length = universal_code_integers(len(new_candidate))
    l_pattern_combination = rulelist.l_variables_in_pattern[len(new_candidate)]
    l_items = sum([rulelist.l_attribute_item[(item.parent_variable, item.number_operators)]
                        for item in new_candidate])
    gain_model = delta_rules - l_pattern_length - l_pattern_combination - l_items
    return gain_model

def compute_delta_score(rulelist, new_candidate, new_subgroup_statistics, new_default_rule_statistics):
    delta_data = compute_delta_data(rulelist, new_subgroup_statistics, new_default_rule_statistics)
    delta_model = compute_delta_model(rulelist, new_candidate)
    usage = new_subgroup_statistics.usage
    delta_score = (delta_data+delta_model) / (usage**rulelist.alpha_gain)
    return delta_score, delta_data, delta_model


def compute_statistics_newrules(rulelist, data, bitarray_subgroup):
    """ Computes the statistics of 3 rules:
        1. the new subgroup rule
        2. the old "default" rule that covered the subgroup (only the cover of the subgroup not the rest)
        3. the new default rule that covers the region not covered by any subgroup.

    """
    rulelist.tmp_subgroup_statistic = rulelist.tmp_subgroup_statistic.replace_stats(data, bitarray_subgroup)
    bitarray_new_defaultrule  =rulelist.bitset_uncovered &~ bitarray_subgroup
    rulelist.tmp_default_statistic = rulelist.tmp_default_statistic.replace_stats(data, bitarray_new_defaultrule)
    return rulelist.tmp_subgroup_statistic, rulelist.tmp_default_statistic