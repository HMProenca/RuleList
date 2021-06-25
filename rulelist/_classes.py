# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:13:18 2019

@author: gathu
"""
from abc import ABCMeta
from abc import abstractmethod
from time import time
from typing import AnyStr

import numpy as np
from sklearn.base import MultiOutputMixin, BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier

from rulelist.rulelistmodel.categoricalmodel.categoricalrulelist import CategoricalRuleList
from rulelist.rulelistmodel.gaussianmodel.gaussianrulelist import GaussianRuleList
from rulelist.rulelistmodel.prediction import predict_rulelist, predict_prob_rulelist
from rulelist.search.iterative_rule_search import greedy_and_beamsearch
from rulelist.util.bitset_operations import bitset2indexes
from rulelist.datastructure.data import Data

class BaseRuleList(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,*,max_depth, beam_width, min_support, n_cutpoints, discretization = "static",
                 max_rules = np.inf, alpha_gain = 1.0):

        if not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
            raise ValueError("max_depth incorrectly selected, please select a "
                             "positive integer greater or equal to 1.")

        if not isinstance(beam_width, (int, np.integer)) or beam_width < 1:
            raise ValueError("beam_width incorrectly selected, please select a "
                             "positive integer greater or equal to 1.")

        if not isinstance(n_cutpoints, (int, np.integer)) or n_cutpoints < 2:
            raise ValueError("n_cutpoints incorrectly selected, please select a "
                             "positive integer greater or equal to 2.")

        if discretization not in ["static","dynamic"]:
            raise ValueError("At this moment we only support \"static\" or \"dynamic\" discretizations.")

        if not isinstance(n_cutpoints, (int, np.integer)) or max_rules < 0:
            raise ValueError("max_rules incorrectly selected, please select a "
                             "zero or a positive integer.")

        if alpha_gain < 0 or alpha_gain > 1:
            raise ValueError("alpha_gain incorrectly selected, please select a "
                             "between zero and 1 inclusive.")

        self.alpha_gain = alpha_gain
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_support = min_support
        self.n_cutpoints = n_cutpoints
        self.discretization = discretization
        self.number_rules = 0
        self.max_rules = max_rules
        self._rulelist = None

    #TODO:  def __repr__
    def __str__(self):
        text2print = self._rulelist.description if self.number_rules > 0 else "Model not fitted"
        return text2print

    def fit(self,X,Y):
        """Fit the model according to the given training datastructure.
        Parameters
        ----------
        df : pandas dataframe with name variables with last column as target
        variable.
        Returns
        -------
        self : object
        """
        is_nominal_target  = is_classifier(self)
        start_time = time()
        #self._rulelist = _fit_rulelist(
        #        X,Y, self.target_model, self.max_depth,self.beam_width,self.min_support, self.n_cutpoints,
        #        self.task,self.discretization,self.max_rules,self.alpha_gain)

        data = Data(input_data=X, n_cutpoints=self.n_cutpoints, discretization=self.discretization,
                    target_data=Y, target_model=self.target_model, min_support=self.min_support)

        if is_nominal_target:
            self._rulelist = CategoricalRuleList(data, self.task, self.max_depth, self.beam_width, self.min_support, self.max_rules,
                                                       self.alpha_gain)
        else:
            self._rulelist = GaussianRuleList(data, self.task, self.max_depth, self.beam_width, self.min_support, self.max_rules,
                                                       self.alpha_gain)
        self._rulelist = greedy_and_beamsearch(data, self._rulelist)
        self._rulelist.add_description()
        self.runtime = time() - start_time
        self.number_rules = self._rulelist.number_rules
        self.rule_sets = [bitset2indexes(bitset) for bitset in self._rulelist.bitset_rules]

        return self


    def predict(self,X):
        """ Predicts the target variable for an input data X.
        ----------
        X : a numpy array or pandas dataframe with the variables in the same
            poistion (column number) as given in "fit" function.

        Returns a numpy array y with the predicted values according to the
        fitted rule list (obtained using the "fit" function above). y has the
        same length as X.shape[0] (number of rows).
        -------
        self : object
        """
        y_hat = predict_rulelist(X, self)
        return y_hat


class RuleListClassifier(ClassifierMixin, BaseRuleList):
    """A probabilistic rule list classifier.

     It can be applied for classification of univariate or multivariate (independent) targets.
     It uses an Minimum Description Length (MDL) formulation to define an optimum rule list.
     For search it resorts combination of greedy search to add one rule at the time, together with beam search to
     find the the rules to add.
     The algorithm is a mixture of [1],[2],[3]. The MDL nominal and numeric encoding, and algorithm is the one
     proposed in [3] for subgroup list discovery

     Parameters
     ----------
     max_depth : int, optional (default=5)
         defines the maximum size that rule description can take based
         on the number of variables that the beam search accepts to refine.
         For example, if 'max_depth = 4' the maximum size of a pattern found is
         4.

     beam_width : int, optional (default=100)
         defines the width of the beam in the beam search, i.e., the number of
         patterns that are selected at each iteration to be expanded.

     min_support : int or float
         defines the minimum support that a rule/subgroup can cover in the training datastructure.
         if positive int, it defines an absolute value.
         if smaller than one float, it defines a relative value, i.e., min_support*number_instances_data.

     n_cutpoints : int, optional (default=5)
        number of cut points used to discretize a single-numeric attribute/variable.
        Note 1: this algorithm creates for each cutpoint a binary split, and
        the combination of all cutpoints. As an example of the former, if the
        cut point is x_cut = 5, it will create both the condition x<5 and x>5.
        In relation to the latter, if two of the cut points are x_cut1=3, and
        x_cut2=5, it will also create  3<x<5.

     discretization : string (default="static")
        (possible values: "static" or "dynamic")
        - "static" - performs a priori discretization of all single-numeric variables
        - "dynamic" - at each iteration of the beam search it conditionally
        discretizes all single-numeric variables based on the given pattern.
        This influences the MDL encoding of the pattern. If static it will use a combination (as order does not matter)
        and if "dynamic" it uses k-permutations of n.

     max_rules : int, optional (default=0)
        Maximum number of subgroups/rules to mine. If max_rules=0 is given it
        continues finding subgroups/rules until no more compression is achieved.

     alpha_gain : int, optional (default=1.0 which equals the "normalized gain" in the literature.)
        (possible values: "absolute" or "normalized")
        Type of score used to expand the beam search and to add a rule/subgroup
        at each iteration.
        - "1.0" - is the "normalized gain" which favors more rules that cover less data.
        - 0.0 - is the "absolute gain" which favors less rules that cover more data.
        All values in ]0,1[ are trade-off between both gains.

     Attributes
     ----------
    number_rules: int
         Number of rules of the list excluding the default rule.

    antecedent_description: list of strings
         string of each rule antecedent description.

    runtime: float
        time in seconds that it took to run the code.

    rule_sets: List of ndarrays
        contains indexes of instances covered by each isolated rule. Note that this does not take into account the
        overlap of the rule list, and thus it only gives the indexes of the association rule
        (as if it was not in a rule list).
        If you want the rulelist ids just do:
        list_sets = [set(idxs) - set().union(*model.rule_sets[:i]) for i, idxs in enumerate(model.rule_sets)]

    _rulelist: RuleSet class
        It contains all the intermediate computations necessary to obtain the output RuleList
        self.runtime = time() - start_time
        self.number_rules = self._rulelist.number_rules
        self.rule_sets = [bitset2indexes(bitset) for bitset in self._rulelist.bitset_rules]

    References:
        .. [1] Proença, Hugo M., and Matthijs van Leeuwen. "Interpretable multiclass classification
               by MDL-based rule lists." Information Sciences 2020.
               https://arxiv.org/abs/1905.00328

        .. [2] Proença H.M., Grünwald P., Bäck T., Leeuwen M.. (2021) Discovering Outstanding Subgroup Lists
               for Numeric Targets Using MDL. ECML PKDD 2020. Springer
               https://arxiv.org/abs/2006.09186

        .. [3] Proença, Hugo Manuel, Thomas Bäck, and Matthijs van Leeuwen. "Robust subgroup discovery."
               arXiv preprint arXiv:2103.13686 (2021).
               https://arxiv.org/abs/2103.13686
     """
    def __init__(self,*,target_model = "categorical", task = "prediction",max_depth=5, beam_width = 100,
                 min_support = 1, n_cutpoints = 5, discretization = "static",
                 max_rules = np.inf, alpha_gain = 1.0):

        super().__init__(max_depth = 5,
                         beam_width = 100,
                         min_support = 1,
                         n_cutpoints = 5,
                         discretization = "static",
                         max_rules = np.inf,
                         alpha_gain = 1.0
        )
        self.task = "prediction"
        self.target_model = "categorical"

    def fit(self, X, y):

        super().fit(
            X,
            y,
        )
        return self

    def predict_proba(self,X):
        """ Returns the probabilities of the target variables for an input data X.
        ----------
        X : a numpy array or pandas dataframe with the variables in the same
            poistion (column number) as given in "fit" function.

        Returns a numpy array prob_hat (univariate case) or a dictionary of numpy arrays
        with the probabilites according to thefitted rule list (obtained using the "fit" function above). y has the
        same length as X.shape[0] (number of rows).
        -------
        self : object
        """
        prob_hat = predict_prob_rulelist(X, self)
        return prob_hat

class RuleListRegressor(RegressorMixin, BaseRuleList):
    """A probabilistic rule list classifier.

     It can be applied for regression of univariate or multivariate (independent) targets.
     It uses an Minimum Description Length (MDL) formulation to define an optimum rule list.
     For search it resorts combination of greedy search to add one rule at the time, together with beam search to
     find the the rules to add.

     The algorithm is a mixture of [1],[2],[3]. The MDL nominal and numeric encoding, and algorithm is the one
     proposed in [3] for subgroup list discovery

     Parameters
     ----------
     max_depth : int, optional (default=5)
         defines the maximum size that rule description can take based
         on the number of variables that the beam search accepts to refine.
         For example, if 'max_depth = 4' the maximum size of a pattern found is
         4.

     beam_width : int, optional (default=100)
         defines the width of the beam in the beam search, i.e., the number of
         patterns that are selected at each iteration to be expanded.

     min_support : int or float
         defines the minimum support that a rule/subgroup can cover in the training datastructure.
         if positive int, it defines an absolute value.
         if smaller than one float, it defines a relative value, i.e., min_support*number_instances_data.

     n_cutpoints : int, optional (default=5)
        number of cut points used to discretize a single-numeric attribute/variable.
        Note 1: this algorithm creates for each cutpoint a binary split, and
        the combination of all cutpoints. As an example of the former, if the
        cut point is x_cut = 5, it will create both the condition x<5 and x>5.
        In relation to the latter, if two of the cut points are x_cut1=3, and
        x_cut2=5, it will also create  3<x<5.

     discretization : string (default="static")
        (possible values: "static" or "dynamic")
        - "static" - performs a priori discretization of all single-numeric variables
        - "dynamic" - at each iteration of the beam search it conditionally
        discretizes all single-numeric variables based on the given pattern.
        This influences the MDL encoding of the pattern. If static it will use a combination (as order does not matter)
        and if "dynamic" it uses k-permutations of n.

     max_rules : int, optional (default=0)
        Maximum number of subgroups/rules to mine. If max_rules=0 is given it
        continues finding subgroups/rules until no more compression is achieved.

     alpha_gain : int, optional (default=1.0 which equals the "normalized gain" in the literature.)
        (possible values: "absolute" or "normalized")
        Type of score used to expand the beam search and to add a rule/subgroup
        at each iteration.
        - "1.0" - is the "normalized gain" which favors more rules that cover less data.
        - 0.0 - is the "absolute gain" which favors less rules that cover more data.
        All values in ]0,1[ are trade-off between both gains.

     Attributes
     ----------
    number_rules: int
         Number of rules of the list excluding the default rule.

    antecedent_description: list of strings
         string of each rule antecedent description.

    runtime: float
        time in seconds that it took to run the code.

    rule_sets: List of ndarrays
        contains indexes of instances covered by each isolated rule. Note that this does not take into account the
        overlap of the rule list, and thus it only gives the indexes of the association rule
        (as if it was not in a rule list).
        If you want the rulelist ids just do:
        list_sets = [set(idxs) - set().union(*model.rule_sets[:i]) for i, idxs in enumerate(model.rule_sets)]

    _rulelist: RuleSet class
        It contains all the intermediate computations necessary to obtain the output RuleList
        self.runtime = time() - start_time
        self.number_rules = self._rulelist.number_rules
        self.rule_sets = [bitset2indexes(bitset) for bitset in self._rulelist.bitset_rules]

    References:
        .. [1] Proença, Hugo M., and Matthijs van Leeuwen. "Interpretable multiclass classification
               by MDL-based rule lists." Information Sciences 2020.
               https://arxiv.org/abs/1905.00328

        .. [2] Proença H.M., Grünwald P., Bäck T., Leeuwen M.. (2021) Discovering Outstanding Subgroup Lists
               for Numeric Targets Using MDL. ECML PKDD 2020. Springer
               https://arxiv.org/abs/2006.09186

        .. [3] Proença, Hugo Manuel, Thomas Bäck, and Matthijs van Leeuwen. "Robust subgroup discovery."
               arXiv preprint arXiv:2103.13686 (2021).
               https://arxiv.org/abs/2103.13686

     """
    def __init__(self,*,max_depth=5, beam_width = 100,
                 min_support = 1, n_cutpoints = 5, discretization = "static",
                 max_rules = np.inf, alpha_gain = 1.0):

        super().__init__(max_depth = 5,
                         beam_width = 100,
                         min_support = 1,
                         n_cutpoints = 5,
                         discretization = "static",
                         max_rules = np.inf,
                         alpha_gain = 1.0
        )
        self.task = "prediction"
        self.target_model = "gaussian"

    def fit(self, X, y):

        super().fit(
            X,
            y,
        )
        return self

class SubgroupListCategorical(ClassifierMixin, BaseRuleList):
    """A probabilistic subgroup list for nominal targets.

     It can be applied for find subgroup lists in data with univariate or multivariate (independent) targets.
     It uses an Minimum Description Length (MDL) formulation to define an optimum subgroup list.
     For search it resorts combination of greedy search to add one rule at the time, together with beam search to
     find the the rules to add.
     The algorithm is a mixture of [1],[2],[3]. The MDL nominal and numeric encoding, and algorithm is the one
     proposed in [3] for subgroup list discovery

     Parameters
     ----------
     max_depth : int, optional (default=5)
         defines the maximum size that rule description can take based
         on the number of variables that the beam search accepts to refine.
         For example, if 'max_depth = 4' the maximum size of a pattern found is
         4.

     beam_width : int, optional (default=100)
         defines the width of the beam in the beam search, i.e., the number of
         patterns that are selected at each iteration to be expanded.

     min_support : int or float
         defines the minimum support that a rule/subgroup can cover in the training datastructure.
         if positive int, it defines an absolute value.
         if smaller than one float, it defines a relative value, i.e., min_support*number_instances_data.

     n_cutpoints : int, optional (default=5)
        number of cut points used to discretize a single-numeric attribute/variable.
        Note 1: this algorithm creates for each cutpoint a binary split, and
        the combination of all cutpoints. As an example of the former, if the
        cut point is x_cut = 5, it will create both the condition x<5 and x>5.
        In relation to the latter, if two of the cut points are x_cut1=3, and
        x_cut2=5, it will also create  3<x<5.

     discretization : string (default="static")
        (possible values: "static" or "dynamic")
        - "static" - performs a priori discretization of all single-numeric variables
        - "dynamic" - at each iteration of the beam search it conditionally
        discretizes all single-numeric variables based on the given pattern.
        This influences the MDL encoding of the pattern. If static it will use a combination (as order does not matter)
        and if "dynamic" it uses k-permutations of n.

     max_rules : int, optional (default=0)
        Maximum number of subgroups/rules to mine. If max_rules=0 is given it
        continues finding subgroups/rules until no more compression is achieved.

     alpha_gain : int, optional (default=1.0 which equals the "normalized gain" in the literature.)
        (possible values: "absolute" or "normalized")
        Type of score used to expand the beam search and to add a rule/subgroup
        at each iteration.
        - "1.0" - is the "normalized gain" which favors more rules that cover less data.
        - 0.0 - is the "absolute gain" which favors less rules that cover more data.
        All values in ]0,1[ are trade-off between both gains.

     Attributes
     ----------
    number_rules: int
         Number of rules of the list excluding the default rule.

    antecedent_description: list of strings
         string of each rule antecedent description.

    runtime: float
        time in seconds that it took to run the code.

    rule_sets: List of ndarrays
        contains indexes of instances covered by each isolated rule. Note that this does not take into account the
        overlap of the rule list, and thus it only gives the indexes of the association rule
        (as if it was not in a rule list).
        If you want the rulelist ids just do:
        list_sets = [set(idxs) - set().union(*model.rule_sets[:i]) for i, idxs in enumerate(model.rule_sets)]

    _rulelist: RuleSet class
        It contains all the intermediate computations necessary to obtain the output RuleList
        self.runtime = time() - start_time
        self.number_rules = self._rulelist.number_rules
        self.rule_sets = [bitset2indexes(bitset) for bitset in self._rulelist.bitset_rules]

    References:
        .. [1] Proença, Hugo M., and Matthijs van Leeuwen. "Interpretable multiclass classification
               by MDL-based rule lists." Information Sciences 2020.
               https://arxiv.org/abs/1905.00328

        .. [2] Proença H.M., Grünwald P., Bäck T., Leeuwen M.. (2021) Discovering Outstanding Subgroup Lists
               for Numeric Targets Using MDL. ECML PKDD 2020. Springer
               https://arxiv.org/abs/2006.09186

        .. [3] Proença, Hugo Manuel, Thomas Bäck, and Matthijs van Leeuwen. "Robust subgroup discovery."
               arXiv preprint arXiv:2103.13686 (2021).
               https://arxiv.org/abs/2103.13686
     """

    def __init__(self, *, max_depth=5, beam_width=100,
                 min_support=1, n_cutpoints=5, discretization="static",
                 max_rules=np.inf, alpha_gain=1.0):
        super().__init__(max_depth=5,
                         beam_width=100,
                         min_support=1,
                         n_cutpoints=5,
                         discretization="static",
                         max_rules=np.inf,
                         alpha_gain=1.0
                         )
        self.task = "discovery"
        self.target_model = "categorical"

    def fit(self, X, y):
        super().fit(
            X,
            y,
        )
        return self

    def predict_proba(self, X):
        """ Returns the probabilities of the target variables for an input data X.
        ----------
        X : a numpy array or pandas dataframe with the variables in the same
            poistion (column number) as given in "fit" function.

        Returns a numpy array prob_hat (univariate case) or a dictionary of numpy arrays
        with the probabilites according to thefitted rule list (obtained using the "fit" function above). y has the
        same length as X.shape[0] (number of rows).
        -------
        self : object
        """
        prob_hat = predict_prob_rulelist(X, self)
        return prob_hat


class SubgroupListGaussian(RegressorMixin, BaseRuleList):
    """A probabilistic subgroup list for numeric targets (modelled with a normal distribution).

     It can be applied for regression of univariate or multivariate (independent) targets.
     It uses an Minimum Description Length (MDL) formulation to define an optimum subgroup list.
     For search it resorts combination of greedy search to add one subgroup at the time, together with beam search to
     find the the subgroup to add.

     The algorithm is a mixture of [1],[2],[3]. The MDL nominal and numeric encoding, and algorithm is the one
     proposed in [3] for subgroup list discovery

     Parameters
     ----------
     max_depth : int, optional (default=5)
         defines the maximum size that rule/subgroup description can take based
         on the number of variables that the beam search accepts to refine.
         For example, if 'max_depth = 4' the maximum size of a pattern found is
         4.

     beam_width : int, optional (default=100)
         defines the width of the beam in the beam search, i.e., the number of
         patterns that are selected at each iteration to be expanded.

     min_support : int or float
         defines the minimum support that a rule/subgroup can cover in the training datastructure.
         if positive int, it defines an absolute value.
         if smaller than one float, it defines a relative value, i.e., min_support*number_instances_data.

     n_cutpoints : int, optional (default=5)
        number of cut points used to discretize a single-numeric attribute/variable.
        Note 1: this algorithm creates for each cutpoint a binary split, and
        the combination of all cutpoints. As an example of the former, if the
        cut point is x_cut = 5, it will create both the condition x<5 and x>5.
        In relation to the latter, if two of the cut points are x_cut1=3, and
        x_cut2=5, it will also create  3<x<5.

     discretization : string (default="static")
        (possible values: "static" or "dynamic")
        - "static" - performs a priori discretization of all single-numeric variables
        - "dynamic" - at each iteration of the beam search it conditionally
        discretizes all single-numeric variables based on the given pattern.
        This influences the MDL encoding of the pattern. If static it will use a combination (as order does not matter)
        and if "dynamic" it uses k-permutations of n.

     max_rules : int, optional (default=0)
        Maximum number of subgroups/rules to mine. If max_rules=0 is given it
        continues finding subgroups/rules until no more compression is achieved.

     alpha_gain : int, optional (default=1.0 which equals the "normalized gain" in the literature.)
        (possible values: "absolute" or "normalized")
        Type of score used to expand the beam search and to add a rule/subgroup
        at each iteration.
        - "1.0" - is the "normalized gain" which favors more rules/subgroups that cover less data.
        - 0.0 - is the "absolute gain" which favors less rules/subgroups that cover more data.
        All values in ]0,1[ are trade-off between both gains.

     Attributes
     ----------
    number_rules: int
         Number of rules/subgroups of the list excluding the default rule.

    antecedent_description: list of strings
         string of each rule/subgroup antecedent description.

    runtime: float
        time in seconds that it took to run the code.

    rule_sets: List of ndarrays
        contains indexes of instances covered by each isolated rule/subgroup. Note that this does not take into account the
        overlap of the rule list, and thus it only gives the indexes of the association rule
        (as if it was not in a rule list).
        If you want the rulelist ids just do:
        list_sets = [set(idxs) - set().union(*model.rule_sets[:i]) for i, idxs in enumerate(model.rule_sets)]

    _rulelist: RuleSet class
        It contains all the intermediate computations necessary to obtain the output RuleList
        self.runtime = time() - start_time
        self.number_rules = self._rulelist.number_rules
        self.rule_sets = [bitset2indexes(bitset) for bitset in self._rulelist.bitset_rules]

    References:
        .. [1] Proença, Hugo M., and Matthijs van Leeuwen. "Interpretable multiclass classification
               by MDL-based rule lists." Information Sciences 2020.
               https://arxiv.org/abs/1905.00328

        .. [2] Proença H.M., Grünwald P., Bäck T., Leeuwen M.. (2021) Discovering Outstanding Subgroup Lists
               for Numeric Targets Using MDL. ECML PKDD 2020. Springer
               https://arxiv.org/abs/2006.09186

        .. [3] Proença, Hugo Manuel, Thomas Bäck, and Matthijs van Leeuwen. "Robust subgroup discovery."
               arXiv preprint arXiv:2103.13686 (2021).
               https://arxiv.org/abs/2103.13686

     """

    def __init__(self, *, max_depth=5, beam_width=100,
                 min_support=1, n_cutpoints=5, discretization="static",
                 max_rules=np.inf, alpha_gain=1.0):
        super().__init__(max_depth=5,
                         beam_width=100,
                         min_support=1,
                         n_cutpoints=5,
                         discretization="static",
                         max_rules=np.inf,
                         alpha_gain=1.0
                         )
        self.task = "discovery"
        self.target_model = "gaussian"

    def fit(self, X, y):
        super().fit(
            X,
            y,
        )
        return self

def RuleList(task = "prediction", target_model = "categorical",max_depth=5, beam_width=100,min_support=1, n_cutpoints=5,
             discretization="static", max_rules=np.inf, alpha_gain=1.0):
    """ This function exist for LEGACY reasons. please avoid using it.

    This function mimics the old RuleList class of package versions 0.X.X

    Parameters
    ----------
    task
    target_model
    max_depth
    beam_width
    min_support
    n_cutpoints
    discretization
    max_rules
    alpha_gain

    Returns
    -------

    """
    if task == "prediction" and target_model == "categorical":
        return RuleListClassifier(max_depth=5, beam_width=100,min_support=1, n_cutpoints=5,
             discretization="static", max_rules=np.inf, alpha_gain=1.0)
    elif task == "prediction" and target_model == "gaussian":
        return RuleListRegressor(max_depth=5, beam_width=100,min_support=1, n_cutpoints=5,
             discretization="static", max_rules=np.inf, alpha_gain=1.0)
    elif task == "discovery" and target_model == "categorical":
        return SubgroupListCategorical(max_depth=5, beam_width=100,min_support=1, n_cutpoints=5,
             discretization="static", max_rules=np.inf, alpha_gain=1.0)
    elif task == "discovery" and target_model == "gaussian":
        return SubgroupListGaussian(max_depth=5, beam_width=100,min_support=1, n_cutpoints=5,
             discretization="static", max_rules=np.inf, alpha_gain=1.0)