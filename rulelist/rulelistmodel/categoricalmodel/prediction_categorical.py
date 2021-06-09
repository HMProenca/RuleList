import numpy as np

def point_value_categorical(statistics):
    class_labels = np.array([max(count_per_class.keys(), key=(lambda k: count_per_class[k]))
     for varname, count_per_class in statistics.usage_per_class.items()])
    return class_labels

def probability_categorical(statistics,target):
    """ Computes the probability with laplace smoothing.

    Adds a little pseudocount (epsilon) which makes for a more balanced probability.
    An epsilon of 0.5 is the Jeffrey's prior for Dirichlet's distribution, and an epsilon of 1 is the uniform prior.

    :param statistics:
    :param target:
    :return:
    """
    usage = statistics.usage
    n_classes = statistics.number_classes[target]
    epsilon = 0.5
    probabilities = np.array([(usg_cl+epsilon)/(usage+epsilon*n_classes)
                              for usg_cl in statistics.usage_per_class[target].values()])
    return probabilities
