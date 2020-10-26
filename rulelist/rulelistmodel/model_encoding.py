from rulelist.datastructure.attribute.attribute import Attribute
from rulelist.mdl.mdl_base_codes import universal_code_integers_maximum, uniform_code, universal_code_integers


def compute_length_model(rulelist):
    """ Computes code length of the model encoding using

    1. Number Rules - Universal code of integers for number of rules
    2. Number variables per pattern - Universal code of integers for number of attributes in a rule/subgroup.
    3. Combination of variable pairs - Uniform code over the combinations of pairs of variables.
    4. Item in the variable - Universal code of integers (conditional on the maximum number of operators) for the number
        of operators used plus an uniform code for the number of subsets formed with those operators in a variable.
    """
    #l_rules = rulelist.l_universal[rulelist.number_rules]
    l_rules = universal_code_integers(rulelist.number_rules)
    l_patterns_length = 0
    l_patterns_combination = 0
    l_items = 0
    for subgroup in rulelist.subgroups:
        #l_patterns_length += rulelist.l_universal[subgroup.size]
        l_patterns_length +=  universal_code_integers(subgroup.size)
        l_patterns_combination += rulelist.l_combination_pattern[subgroup.size]
        l_items += sum([rulelist.l_attribute_item[(item.parent_variable, item.number_operators)]
                        for item in subgroup.pattern])
    l_model = l_rules + l_patterns_length + l_patterns_combination + l_items
    return l_model


def compute_item_length(attribute: Attribute) -> float:
    """ Computes the code of an attribute based on its cardinality
    """
    for n_operators in range(1,attribute.max_operators+1):
        l_number_operators = universal_code_integers_maximum(n_operators,attribute.max_operators)
        l_code = uniform_code(attribute.cardinality_operator[n_operators])

        l_item = l_number_operators + l_code
        yield attribute.name, n_operators, l_item



#def compute_item_length_uniformforall(attribute: Attribute) -> float:
#    cardinality  = sum([attribute.cardinality_operator[n_operators] for n_operators in range(1,attribute.max_operators+1)])
#    l_item = uniform_code(cardinality)
#    for n_operators in range(1,attribute.max_operators+1):
#        yield attribute.name, n_operators, l_item