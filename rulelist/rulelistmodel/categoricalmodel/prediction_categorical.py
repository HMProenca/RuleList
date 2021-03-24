import numpy as np

def point_value_categorical(statistics):
    class_labels = np.array([max(count_per_class.keys(), key=(lambda k: count_per_class[k]))
     for varname, count_per_class in statistics.usage_per_class.items()])
    return class_labels

def probability_categorical(statistics,target):
    usage = statistics.usage
    probabilities = np.array([usg_cl/usage for usg_cl in statistics.usage_per_class[target].values()])
    return probabilities
