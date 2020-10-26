from functools import reduce

import numpy as np
import pandas as pd

from rulelist.rulelistmodel.categoricalmodel.prediction_categorical import point_value_categorical
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
    # TODO: add probability estimate if the user wants
    probability = np.empty((n_predictions,n_targets),dtype=float)
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