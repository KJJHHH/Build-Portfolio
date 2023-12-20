import copy
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression 

def lr(
        
        fnl_df,                  # data
        model_strategy,          # dict
        longshort_thres = None,  # dict
        tune = False
        ):
