import numpy as np
import pylab as pl
import matplotlib.font_manager
from scipy import stats

from sklearn import grid_search, metrics, \
                preprocessing
from sklearn.ensemble import GradientBoostingClassifier, \
                        RandomForestClassifier

"""We have to implement here our own
learning to rank method.

We must start from a common classification
method. we are going to start with 
RandomForests. 

Assume a three-class problem:
    - non-popular: 0
    - popular: 1
    - high popular: 2
    
For defining ranking in a classification, we
will try to apply a DCG metric.

In the near future, we 
might try Gradient Boosting (GBRT) and
also Extremely Randomized Trees

#documentation about boosting methods:
http://scikit-learn.org/0.11/modules/ensemble.html
#a tutorial about DCG:
http://www.cc.gatech.edu/~kzhou30/learndcg/web.html
"""
