from functools import reduce

import numpy as np
import pandas as pd

from rulelist.rulelistmodel.categoricalmodel.prediction_categorical import point_value_categorical, \
    probability_categorical
from rulelist.rulelistmodel.gaussianmodel.prediction_gaussian import point_value_gaussian
from rulelist.rulelistmodel.rulesetmodel import RuleSetModel

point_value_estimation = {
    "gaussian" : point_value_gaussian,
    "categorical": point_value_categorical
}

def predict_rulelist(X : pd.DataFrame, rulelist: RuleSetModel):
    if X is not pd.DataFrame: Exception('X needs to be a DataFrame')
    n_predictions = X.shape[0]
    n_targets = rulelist.default_rule_statistics.number_targets
    instances_covered = np.zeros(n_predictions, dtype=bool)
    predictions = np.empty((n_predictions,n_targets),dtype=object)
    for subgroup in rulelist.subgroups:
        instances_subgroup = ~instances_covered &\
                             reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        predictions[instances_subgroup,:] = point_value_estimation[rulelist.target_model](subgroup.statistics)
        instances_covered |= instances_subgroup

    # default rule
    predictions[~instances_covered, :] = point_value_estimation[rulelist.target_model](rulelist.default_rule_statistics)
    if n_targets == 1:
        predictions = predictions.flatten()

    # if int values try to return ints
    try:
        predictions = predictions.astype(int)
    except ValueError:
        pass
    return predictions

def predict_prob_rulelist(X : pd.DataFrame, rulelist: RuleSetModel):
    if X is not pd.DataFrame: Exception('X needs to be a DataFrame')
    if rulelist.target_model != 'categorical': Exception('It needs to be a classification setting.')

    n_predictions = X.shape[0]
    n_targets = rulelist.default_rule_statistics.number_targets
    n_classes = [v for v in rulelist.default_rule_statistics.number_classes.values()]
    target_names = [v for v in rulelist.default_rule_statistics.number_classes.keys()]
    instances_covered = np.zeros(n_predictions, dtype=bool)
    probability = {t: np.empty((n_predictions,n_classes[it]),dtype=object)
                      for it,t in enumerate(target_names)}
    for subgroup in rulelist.subgroups:
        instances_subgroup = ~instances_covered &\
                             reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        for t in target_names:
            probability[t][instances_subgroup,:] = probability_categorical(subgroup.statistics,t)
        instances_covered |= instances_subgroup

    # default rule
    for t in target_names:
        probability[t][~instances_covered, :] = probability_categorical(rulelist.default_rule_statistics,t)
    if n_targets == 1:
        probability = probability[target_names[0]]

    return probability