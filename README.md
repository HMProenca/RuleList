

# MDL Rule Lists for prediction and subgroup discovery.

[![PyPI version](https://badge.fury.io/py/rulelist.svg)](https://badge.fury.io/py/rulelist)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rulelist)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for using rule lists for univariate or multivariate classification or regression and its equivalents in Data Mining and Subgroup Discovery. 
These models use the Minimum Description Length (MDL) principle as optimality criteria.


## Dependencies

This project was written for Python 3.7. All required packages from PyPI are specified in the `requirements.txt`.

*NOTE:* This list of packages includes the `gmpy2` package.

## Installation

For the latest version clone this package as is and use it directly:

```bash
$ git clone https://github.com/HMProenca/RuleList
```
For the latest stable version from pip (it can be older than the current github version) please use

```bash
pip install rulelist
```

If you run into issues regarding the `gmpy2` package mentioned above, please refer to their documentation for help.

For the current version, you can clone the repository and install the dependencies locally:

```bash
git clone https://github.com/HMProenca/RuleList.git
cd RuleList
pip install -r requirements.txt
```


## Example of usage for prediction:

```python
import pandas as pd
from rulelist import RuleListClassifier, RuleListRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split


data = datasets.load_breast_cancer()
Y = pd.Series(data.target)
X = pd.DataFrame(data.data)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

model = RuleListClassifier(discretization = "static")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test.values,y_pred)

print(model)
```

## Example of usage for subgroup discovery:

```python
import pandas as pd
from rulelist import SubgroupListCategorical, SubgroupListGaussian
from sklearn import datasets

data = datasets.load_boston()
y = pd.Series(data.target)
X = pd.DataFrame(data.data)

model = SubgroupListGaussian()

model.fit(X, y)

print(model)
```



## Contact

If there are any questions or issues, please contact me by mail at `hugo.manuel.proenca@gmail.com` or open an issue here on Github.


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
and
```
@article{proencca2021robust,
  title={Robust subgroup discovery},
  author={Proen{\c{c}}a, Hugo Manuel and B{\"a}ck, Thomas and van Leeuwen, Matthijs},
  journal={arXiv preprint arXiv:2103.13686},
  year={2021}
}
```

# References #
 * [Interpretable multiclass classification by MDL-based rule lists. Hugo M. Proença, Matthijs van Leeuwen. Information Sciences 512 (2020): 1372-1393.](https://www.sciencedirect.com/science/article/pii/S0020025519310138) or publicly available in [ArXiv](https://arxiv.org/abs/1905.00328) -- experiments code (old version) available [here](https://github.com/HMProenca/MDLRuleLists)
 * [Discovering outstanding subgroup lists for numeric targets using MDL. Hugo M. Proença, Peter Grünwald, Thomas Bäck, Matthijs van Leeuwen. ECML-PKDD(2020): ](https://arxiv.org/abs/2006.09186) -- experiments code available [here](https://github.com/HMProenca/SSDpp-numeric)
 * [Robust subgroup discovery. Hugo M. Proença,Thomas Bäck, Matthijs van Leeuwen. (2021) ](https://arxiv.org/abs/2103.13686) -- experiments code available [here](https://github.com/HMProenca/RobustSubgroupDiscovery)
