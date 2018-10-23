# import numpy as np
from scipy.stats.mstats import zscore
import time
# import warnings
# from math import sqrt
# from random import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.externals import joblib
# import collections
# import pandas as pd
# import matplotlib
# from matplotlib import pyplot as plt
# import pickle
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.externals import joblib
# from sklearn import linear_model
# import seaborn as sns
#
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
import pandas as pd
# from math import sqrt
import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression  # L2
from sklearn.ensemble import RandomForestClassifier as RF  # random forests
from sklearn import svm # svm
#import xgboost as xgb  # xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut # leave one group out
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RandomizedLogisticRegression as RL
from sklearn. preprocessing import minmax_scale
#
#
# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
# import hyperopt
# #import xgboost as xgb
import collections
from functools import wraps

#from noisy_classifier_class import*


# timer function
def timethis(func):
    '''
    Decorator that reports execuation time
    :param func: function
    :return: wrapper function
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper



def select_phase(dataset, phase = 'WORD'):

    X = dataset['X']
    y = dataset['y']
    listpos = dataset['list']
    session = dataset['session']
    event_type = dataset['type']

    dataset_select = collections.OrderedDict()
    if (phase == 'ALL'):
        return dataset
    else:
        indices = np.where(event_type == phase)[0]
        dataset_select['X'] = X[indices,:]
        dataset_select['y'] = y[indices]
        dataset_select['list'] = listpos[indices]
        dataset_select['session'] = session[indices]
        dataset_select['type'] = event_type[indices]
        return dataset_select




def get_sample_weights_fr(y):

    n = len(y)
    weights = np.zeros(n)
    recall_mask = y == 1;
    n_recall = np.sum(recall_mask)
    n_non_recall = n - n_recall
    pos_weight = 1.0*n/n_recall
    neg_weight = 1.0*n/n_non_recall
    weights[recall_mask] = pos_weight
    weights[~recall_mask] = neg_weight

    return weights, pos_weight, neg_weight
