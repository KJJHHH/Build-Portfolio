import copy
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
# -----------------------------------------------
from ..train import summary

def params_setting(model):
    if model == "linear":
        params = {
            # "normalize": True        
        }
    elif model == "decision tree":
        params = { # decision tree
            "criterion": ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],  # squared_e defaulted
            "max_depth": [None, 5, 10], # 
            "min_samples_split": [5, 10],
        }
    elif model == "random forest":
        params = {
            'n_estimators': [20], # 50, 100, 
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    elif model == "xgboost":
        params = {
            "learning_rate": [0.01, 0.1, 0.001],
            "n_estimators": [5, 10, 20, 30], # original set: [5, 10, 20, 30]
            "max_depth": [None, 3, 10, 5],
            "min_child_weight": [1, 2, 3] 
        }
    elif model == "svm": 
        params = {
            "C": [0.1, 1, 10],
            "kernel": [ "rbf"], # "poly", "linear",
            "gamma": ["scale", "auto", 0.1, 1]
        }
    elif model == "neural network": # tune in random, so only set the n_trials of tune
        params = {
            "batch_size": 25,
            }
    elif model == "elastic net":
        params = {
            "l1_ratio" : np.arange(0., 1., .1),
            "alpha" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        }
    elif model == "ensemble":
        params = {}
    return params

def ensemble_model(params_setting, fnl_df, model_strategy, longshort_thres, tune):
    # models for ensemble
    models = ["linear", "elastic net", "decision tree", "random forest", "xgboost", "svm"]
    model_strategy["long"] = 20
    model_strategy["short"] = 0
    cross_model = pd.DataFrame(
        columns=["code", "ymd", "return", "prediction", "per_long", "per_short", "pred_ls", "weight"])
    # =========================================
    # implement differnet models
    # some temp datas info:
    #   true predict columns: return, performance, per_long (long decision), per_short (short decision)
    #   rplsw: return, performance, per_long, per_short, weight
    #   longshort_threshold: dict of {"long": long, " short": short}
    for model in models:
        print(model)
        model_strategy["model"] = model
        param = params_setting(model)
        longshort_thres = None
        t = summary(param, input_size=89)
        true_predict, longshort_thres, loss = t.train_lsdecision(
            fnl_df, 
            model_strategy, 
            longshort_thres, # not useful here: == None. decide long/short/nothing with frequency
            tune
            )
        performance, rplsw, market = t.compute_ret(
            data_tp=true_predict, 
            not_short=True, 
            print_detail=False)
        cross_model = pd.concat([cross_model, rplsw[rplsw["pred_ls"] == 1].reset_index()], axis = 0)
    counts = cross_model["code"].value_counts() # chatgpt chat named Dataframe 
    high_prediction = counts[counts >= 5].index
    cross_model["f"] = cross_model["code"].map(counts)
    cross_model = cross_model.drop_duplicates(subset=["code"])
    cross_model = cross_model[cross_model["code"].isin(high_prediction)]
    cross_model["weight"] = cross_model["f"]/cross_model["f"].sum()
    true_predict = cross_model
    print(f"Check weight ensemble: {true_predict['weight'].sum()}")
    performance = pd.DataFrame([(true_predict["weight"]*true_predict["return"]).sum()], 
                               columns=["performance"])
    performance["ymd"] = true_predict["ymd"].unique()[0]
    return performance, market, rplsw