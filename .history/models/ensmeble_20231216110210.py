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

def ensemble_model(t, fnl_df, model_strategy, longshort_thres, tune):
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