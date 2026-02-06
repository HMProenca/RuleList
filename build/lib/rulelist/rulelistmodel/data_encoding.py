from rulelist.rulelistmodel.categoricalmodel.mdl_categorical import length_rule_free_categorical
from rulelist.rulelistmodel.gaussianmodel.mdl_gaussian import length_rule_free_gaussian

length_rule_free = {
    "gaussian": length_rule_free_gaussian,
    "categorical": length_rule_free_categorical
}

def compute_length_data(rulelist):
    """ Computes the code length of the whole rule list.
    """
    l_data_subgroups = sum([length_rule_free[rulelist.target_model](rulelist, subgroup.statistics)
                            for subgroup in rulelist.subgroups])
    l_data = l_data_subgroups + rulelist.length_defaultrule
    return l_data

