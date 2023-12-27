import copy
import gc
import pickle
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
# ------------------------
from models.linear_regression import linear_regression
from models.elastic_net import elastic_net
from models.decision_tree import decision_tree
from models.xgboost import xgboost
from models.random_forest import random_forest
from models.svm import svm
from models.deep_learning import deep_learning

gc.collect()

class summary():
    #####################################################################
    """
    === WHAT IS DOING HERE ===
    Build the pipeline of data and train in SINGLE period (month),
    which include functions:
    0. Initialise:
        train and test date
    1. traintes_period:
        get the train and test period of date, for initialise or update
    2. Xy:
        Split data into X_train, y_train, X_test, y_test
    3. train_lsdecision:
        Train the model and decide the long and short boundary
    4. compute_ret:
        Compute the return for this period
    """

    ######################################################################
    def __init__(self, 
                model_name,
                data,
                industry,
                train_size, 
                test_size, 
                test_start, 
                long_bound = 80,  # percentile
                short_bound = 20,  # percentile
                ls_decision = ["test", "no running"], 
                n_trials = 1, 
                tune = False, 
                short = True,
                input_size = 89):
        
        """ parameters:
        model_name: str, "linear", ...
        data: data of all 
        train_size: year
        test_size: month
        test_start: decide backtest start date (input year, from Jan)
        long_bound:
        short_bound:
        ls_decision:
        n_trial:
        tune:
        short: if short or not
        input_size
        """
        super(summary, self).__init__()
        self.model_name = model_name            
        self.data = data
        self.param = self.params_setting()
        self.input_size = input_size

        self.industry = industry
        self.train_size = train_size
        self.test_size = test_size
        self.test_start = \
            pd.Timestamp(year = test_start, month = 1, day = 1)
        (
        self.train_start, 
        self.test_end) = self.initialise()

        self.long_bound = long_bound
        self.short_bound = short_bound
        self.ls_decision = ls_decision
        # ls_bound
        #   if predicted returns > self.ls_bound["long"]: long
        #   if predicted returns < self.ls_bound["short"]: short
        self.ls_bound = None                
        self.do_short = short

        self.n_trials = n_trials
        self.tune = tune

        self.dir = os.getcwd()
        self.train_test_X_y = None

    #
    def build_portfolio(self):

        splited_data = self.Xy()

        if self.check_sample(splited_data) == False:
            return None

        if self.check_test_date(splited_data):
            (train_hat, test_hat, loss) = self.train(splited_data)

            # 1. data_tp: data with true y and predicted y
            # 2. ls_bound: long and short boundary
            (data_tp, ls_bound) = self.ls_decide(train_hat, test_hat)

            # 1. performance: portfolio returns
            # 2. rplsw: returns, predicted, long_bound, short_bound, weight
            # 3. market: market returns
            (performance, rplsw, market) = self.compute_ret(data_tp, print_detail=False)

            return (performance, rplsw, market)
        
        else:
            return "Test data > 1"

    # 
    def initialise(self):
        train_start = self.test_start - pd.DateOffset(years=self.train_size)
        test_end = self.test_start + pd.DateOffset(months=self.test_size)
        return train_start, test_end
        # self.test_start
        
    # get start and end date for train and test and future predict
    def train_test_update(self): 
        self.train_start = self.train_start + pd.DateOffset(months=self.test_size)
        self.test_start = self.test_start + pd.DateOffset(months=self.test_size)
        self.test_end = self.test_end + pd.DateOffset(months=self.test_size)
    
    # X_train, test, y_train, test
    def Xy(self): 
        temp_d = copy.copy(self.data)
        d_train = temp_d.reset_index()[
            (temp_d.reset_index()["ymd"] >= self.train_start) & \
            (temp_d.reset_index()["ymd"] < self.test_start)]\
            .set_index(["code", "ymd"])
        d_test = temp_d.reset_index()[
            (temp_d.reset_index()["ymd"] >= self.test_start) & \
            (temp_d.reset_index()["ymd"] <= self.test_end)]\
            .set_index(["code", "ymd"])
        X_train = d_train.drop(["return"], axis = 1)
        y_train = d_train["return"]
        X_test = d_test.drop(["return"], axis = 1)
        y_test = d_test["return"]
        return (X_train, y_train, X_test, y_test)
    
    # Check
    def check_test_date(splited_data):    
        if len(splited_data[2].reset_index().groupby("ymd").count()) != 1: 
            print("/"*70)
            print("Weird data: too many dates in test data, expect 1!")
            print(splited_data[2]) 
            "Weird data: too many dates in test data, expect 1!"
            return False
        return True
    def check_sample(splited_data):            
        if splited_data[0].shape[0] <= splited_data[0].shape[1]:
            print(f"Warning: Sample size too small -> size {splited_data[0].shape}")   
            return False
        return True

    # hyperparam
    def params_setting(self):
        if self.model_name == "linear":
            params = {
                # "normalize": True        
            }
        elif self.model_name == "decision tree":
            params = { # decision tree
                "criterion": ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],  # squared_e defaulted
                "max_depth": [None, 5, 10], # 
                "min_samples_split": [5, 10],
            }
        elif self.model_name == "random forest":
            params = {
                'n_estimators': [20], # 50, 100, 
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        elif self.model_name == "xgboost":
            params = {
                "learning_rate": [0.01, 0.1, 0.001],
                "n_estimators": [5, 10, 20, 30], # original set: [5, 10, 20, 30]
                "max_depth": [None, 3, 10, 5],
                "min_child_weight": [1, 2, 3] 
            }
        elif self.model_name == "svm": 
            params = {
                "C": [0.1, 1, 10],
                "kernel": ["rbf"], # 跑太久：, "poly", "linear"
                "gamma": ["scale", "auto", 0.1, 1]
            }
        elif self.model_name == "neural network": # tune in random, so only set the n_trials of tune
            params = {
                "batch_size": 25,
                }
        elif self.model_name == "elastic net":
            params = {
                "l1_ratio" : np.arange(0., 1., .1),
                "alpha" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            }
        elif self.model_name == "ensemble":
            params = {}
        return params

    # train
    def train(self, splited_data):        
        ############################################
        # For 1 single peoriod, 
        # train and decide the long and short boundary
        ############################################
        print(f"param: {self.param}")
        
        if self.model_name == "linear": # False: 
            test_hat, train_hat, test_loss = linear_regression(self.param, splited_data, self.tune)
        elif self.model_name == "decision tree": # False: 
            test_hat, train_hat, test_loss = decision_tree(self.param, splited_data, self.tune)
        elif self.model_name == "xgboost": # False: 
            test_hat, train_hat, test_loss = xgboost(self.param, splited_data, self.tune)
        elif self.model_name == "random forest":
            test_hat, train_hat, test_loss = random_forest(self.param, splited_data, self.tune)
        elif self.model_name == "svm": # False:
            test_hat, train_hat, test_loss = svm(self.param, splited_data, self.tune)
        elif self.model_name == "elastic net":
            test_hat, train_hat, test_loss = elastic_net(self.param, splited_data, self.tune)
        elif self.model_name == "neural network":
            dl = deep_learning(self.industry, self.param, self.input_size)
            test_hat, train_hat, test_loss = dl.dl_tuning_pipeline(splited_data, self.n_trials, self.tune)
        
        return train_hat, test_hat, test_loss

    # decide what to long and short
    def ls_decide(self, train_hat, test_hat):

        (X_train, y_trian, X_test, y_test) = self.Xy()
        data_tp = pd.concat([y_test, test_hat], axis = 1)  
        if self.ls_decision[0] == "train":
            ls_decision_by = train_hat 
        else:
            ls_decision_by = test_hat

        # ====================================================
        # For first rolling prediction for all models and 
        # all rolling predictions for ensemble 
        # do not use running mean
        if (self.ls_bound == None) or (self.ls_decision[1] != "running"):   
            self.ls_bound = {}
            self.ls_bound["long"] = np.percentile(ls_decision_by, self.long_bound) 
            self.ls_bound["short"] = np.percentile(ls_decision_by, self.short_bound) 
            data_tp["per_long"] = self.ls_bound["long"] 
            data_tp["per_short"] = self.ls_bound["short"]
            return data_tp, self.ls_bound
        
        # Running mean for the thresholds
        else: 
            self.ls_bound["long"] =  (self.ls_bound["long"]/2 + 
                                        np.percentile(ls_decision_by, self.long_bound))
            self.ls_bound["short"] = (self.ls_bound["short"]/2 +
                                        np.percentile(ls_decision_by, self.short_bound))
            data_tp["per_long"] = self.ls_bound["long"]
            data_tp["per_short"] = self.ls_bound["short"]
            return data_tp, self.ls_bound
        
        # ====================================================

    # backtest return
    def compute_ret(self, data_tp, print_detail = True): # class and returns
        data_tp = pd.DataFrame(data_tp)
        date = data_tp.reset_index()["ymd"].unique()[0]
        # ====================================================
        # get ls
        single_date = data_tp
        if self.do_short == True:
            single_date["pred_ls"] = np.where(single_date["prediction"] > single_date["per_long"], 1,
                np.where(single_date["prediction"] < single_date["per_short"], -1, 0))
        # ====================================================

        # ====================================================
        # since using train data to determined threshold, so if don't 
        # want to short need to run this
        if self.do_short == True:
            single_date["pred_ls"] = np.where(single_date["prediction"] > single_date["per_long"], 1, 0)
                
        # ====================================================
        # weight
        single_date['weight'] = abs(
            single_date["prediction"]/sum(abs(single_date["prediction"]*single_date["pred_ls"])))
        
        # ====================================================
        single_returns = single_date["return"]*single_date["pred_ls"]*single_date["weight"] # compute returns with  different weight
        rplsw = single_date
        market = pd.DataFrame([single_date["return"].mean()], columns = ["return"])
        performance = [single_returns.sum()]
        performance = pd.DataFrame(performance, columns=["performance"])
        performance["ymd"] = [date]
        market["ymd"] = [date]
            
        # ====================================================
        # print
        if print_detail == True:
            # print(f"long stock threshold: {single_date['per_long'][0]})
            print(f"check sum weight:\
                   {abs((single_date['weight']*single_date['pred_ls'])).sum()}")
            print(f"Long: \
                {single_date[single_date['pred_ls'] == 1].drop(['per_long', 'per_short'], axis = 1)}")
            print(f"Short: \
                {single_date[single_date['pred_ls'] == -1].drop(['per_long', 'per_short'], axis = 1)}")
            print(f"Performance: {performance}")

        return performance, rplsw, market # true, pred, long, short, weight
