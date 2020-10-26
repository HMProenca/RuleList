

# MDL Rule Lists for prediction and data mining

This repository contains the code for using rule lists for univariate or multivariate classification or regression and its equivalents in Data Mining and Subgroup Discovery. 
These models use the Minimum Description Length (MDL) principle as optimality criteria.


## Dependencies

This project was written for Python 3.7. All required packages from PyPI are specified in the `requirements.txt`.

*NOTE:* This list of packages includes the `gmpy2` package.


## Installation

You can pip install this package:
```

```

## Example of usage:

```python
import pandas as pd
from rulelists import RuleList
from sklearn import datasets
from sklearn.model_selection import train_test_split

task = 'prediction'
target_type = 'categorical'

data = datasets.load_breast_cancer()
target = pd.Series(data.target)
data = pd.DataFrame(data.data)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.3)

model = RuleList(task = task, target_type = target_type)

model.fit(X_train, y_train)

y_hat = model.predict(X_test)

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

test_preds = sgclassifier.predict(X_test)


```




## Contact

If there are any questions or issues, please come to the [Gitter chatroom](https://gitter.im/pyModEA/configuring-cmaes) or open an issue here on Github.


## Citation

In a machine learning (prediction) context for problems of classification, regression, multi-label classification, multi-category classification, or multivariate regression cite the corresponding bibtex of the first classification application of MDL rule lists:

```
@article{proencca2020interpretable,
  title={Interpretable multiclass classification by MDL-based rule lists},
  author={Proen{\c{c}}a, Hugo M and van Leeuwen, Matthijs},
  journal={Information Sciences},
  volume={512},
  pages={1372--1393},
  year={2020},
  publisher={Elsevier}
}
```

in the context of data mining and subgroup discovery please refer to subgroup lists:

```
@article{proencca2020discovering,
  title={Discovering outstanding subgroup lists for numeric targets using MDL},
  author={Proen{\c{c}}a, Hugo M and Gr{\"u}nwald, Peter and B{\"a}ck, Thomas and van Leeuwen, Matthijs},
  journal={arXiv preprint arXiv:2006.09186},
  year={2020}
} 
```

# References #
 * [Interpretable multiclass classification by MDL-based rule lists. Hugo M. Proença, Matthijs van Leeuwen. Information Sciences 512 (2020): 1372-1393.](https://www.sciencedirect.com/science/article/pii/S0020025519310138) or publicly available in [ArXiv](https://arxiv.org/abs/1905.00328) -- experiments code (old version) available [here](https://github.com/HMProenca/MDLRuleLists)
 * [Discovering outstanding subgroup lists for numeric targets using MDL. Hugo M. Proença,Thomas Bäck, Matthijs van Leeuwen. ECML-PKDD(2020): ](https://arxiv.org/abs/2006.09186) -- experiments code available [here](https://github.com/HMProenca/SSDpp-numeric)