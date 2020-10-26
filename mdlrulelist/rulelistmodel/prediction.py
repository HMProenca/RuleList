from mdlrulelist.rulelistmodel.categoricalmodel.prediction_categorical import point_value_categorical
from mdlrulelist.rulelistmodel.gaussianmodel.prediction_gaussian import point_value_gaussian
from mdlrulelist.rulelistmodel.rulesetmodel import RuleSetModel
import pandas as pd
import numpy as np
from functools import reduce

point_value_estimation = {
    "gaussian" : point_value_gaussian,
    "categorical": point_value_categorical
}

def predict_rulelist(X : pd.DataFrame, rulelist: RuleSetModel):
    if X is not pd.DataFrame: Exception('X needs to be a DataFrame')
    n_predictions = X.shape[0]
    n_targets = rulelist.defaultrule_statistics.number_targets
    instances_covered = np.zeros(n_predictions, dtype=bool)
    predictions = np.empty((n_predictions,n_targets),dtype=object)
    for subgroup in rulelist.subgroups:
        instances_subgroup = ~instances_covered &\
                             reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        predictions[instances_subgroup,:] = point_value_estimation[rulelist.target_model](subgroup.statistics)
        instances_covered |= instances_subgroup

    # default rule
    predictions[~instances_covered, :] = point_value_estimation[rulelist.target_model](rulelist.defaultrule_statistics)
    if n_targets == 1:
        predictions = predictions.flatten()
    return predictions