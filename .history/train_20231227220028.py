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
        self.test_start = pd.Timestamp(year = test_start, month = 1, day = 1)
        (
        self.train_start, 
        self.test_end) = self.initialise()

        self.long_bound = long_bound
        self.short_bound = short_bound
        self.ls_decision = ls_decision
        self.do_short = short

        self.n_trials = n_trials
        self.tune = tune

        self.dir = os.getcwd()
        self.train_test_X_y = None

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
            (temp_d.reset_index()["ymd"] <= self.test_start)]\
            .set_index(["code", "ymd"])
        d_test = temp_d.reset_index()[
            (temp_d.reset_index()["ymd"] > self.test_start) & \
            (temp_d.reset_index()["ymd"] <= self.test_end)]\
            .set_index(["code", "ymd"])
        X_train = d_train.drop(["return"], axis = 1)
        y_train = d_train["return"]
        X_test = d_test.drop(["return"], axis = 1)
        y_test = d_test["return"]
        return (X_train, y_train, X_test, y_test)
    
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
    def train(self):        
        ############################################
        # For 1 single peoriod, 
        # train and decide the long and short boundary
        ############################################
        print(f"param: {self.param}")
        splited_data = self.Xy()
        
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
    def ls_decide(self, y_test, train_hat, test_hat):

        splited_data = self.Xy()
        (train_hat, test_hat, test_loss) = self.train()

        data_tp = pd.concat([y_test, test_hat], axis = 1)  
        if self.ls_decision[0] == "train":
            ls_decision = train_hat 
        else:
            ls_decision = test_hat

        # ====================================================
        # For first rolling prediction for all models and 
        # all rolling predictions for ensemble 
        # do not use running mean
        if (longshort_thres == None) or (self.ls_decision[1] != "running"):   
            longshort_thres = {}
            longshort_thres["long"] = np.percentile(ls_decision, self.long_bound) 
            longshort_thres["short"] = np.percentile(ls_decision, self.short_bound) 
            data_tp["per_long"] = np.percentile(ls_decision, self.long_bound) 
            data_tp["per_short"] = np.percentile(ls_decision, self.short_bound) 
            return data_tp, longshort_thres
        
        # Running mean for the thresholds
        else: 
            longshort_thres["long"] =  (longshort_thres["long"] + 
                                        np.percentile(ls_decision, self.long_bound))/2
            longshort_thres["short"] = (longshort_thres["short"] +
                                        np.percentile(ls_decision, self.short_bound))/2
            data_tp["per_long"] = longshort_thres["long"]
            data_tp["per_short"] = longshort_thres["short"]
            return data_tp, longshort_thres
        
        # ====================================================

    # backtest return
    def compute_ret(self, data_tp, not_short = False, print_detail = True): # class and returns
        data_tp = pd.DataFrame(data_tp)
        date = data_tp.reset_index()["ymd"].unique()[0]
        # ====================================================
        # get ls
        single_date = data_tp
        single_date["pred_ls"] = np.where(
            single_date["prediction"] > single_date["per_long"], 1,
            np.where(single_date["prediction"] < single_date["per_short"], -1, 0))
        # ====================================================

        # ====================================================
        # since using train data to determined threshold, so if don't 
        # want to short need to run this
        if not_short == True:
            single_date["pred_ls"] = np.where(single_date["prediction"] > single_date["per_long"], 1, 0)
        
        
        # ====================================================
        # weight
        single_date['weight'] = abs(
            single_date["prediction"]/sum(abs(single_date["prediction"]*single_date["pred_ls"])))
        
        # ====================================================
        # print
        print(f"Check sum of weight: {(abs(single_date['weight']*single_date['pred_ls'])).sum()}")
        if print_detail == True:
            print(f"check weight {abs((single_date['weight']*single_date['pred_ls'])).sum()}")
            print(f"long stock threshold {single_date['per_long'][0]}\n\
                date {single_date[single_date['pred_ls'] == 1].drop(['per_long', 'per_short'], axis = 1)}")
            print(f"short stock threshold {single_date['per_short'][0]}\n\
                date {single_date[single_date['pred_ls'] == -1].drop(['per_long', 'per_short'], axis = 1)}")
        
        # ====================================================
        single_returns = single_date["return"]*single_date["pred_ls"]*single_date["weight"] # compute returns with  different weight
        rplsw = single_date
        market = pd.DataFrame([single_date["return"].mean()], columns = ["return"])
        performance = [single_returns.sum()]
        performance = pd.DataFrame(performance, columns=["performance"])
        performance["ymd"] = [date]
        market["ymd"] = [date]
            
        return performance, rplsw, market # true, pred, long, short, weight

    
    
    # ======================================================================
    # old version compute ret
    """
    def compute_ret(self, data_tp, not_short = False, print_detail = True): # class and returns
        data_tp = pd.DataFrame(data_tp)
        date = data_tp.reset_index()["ymd"].unique()

        single_date = data_tp.reset_index()[data_tp.reset_index()["ymd"] == date[0]].set_index(["code", "ymd"])
        single_date["pred_ls"] = np.where(
            single_date["prediction"] > single_date["per_long"], 1,
            np.where(single_date["prediction"] < single_date["per_short"], -1, 0))
        
        ################################################################
        # since using train data to determined threshold, so if don't 
        # want to short need to run this
        if not_short == True:
            single_date["pred_ls"] = np.where(single_date["prediction"] > single_date["per_long"], 1, 0)
        ################################################################
        
        ################################################################
        # weight
        single_date['weight'] = abs(
            single_date["prediction"]/sum(abs(single_date["prediction"]*single_date["pred_ls"])))
        ################################################################

        ################################################################
        # print
        print(f"Check sum of weight: {(single_date['weight']*single_date['pred_ls']).sum()}")
        if print_detail == True:
            print(f"check weight {abs((single_date['weight']*single_date['pred_ls'])).sum()}")
            print(f"long stock threshold {single_date['per_long'][0]}\n\
                date {single_date[single_date['pred_ls'] == 1].drop(['per_long', 'per_short'], axis = 1)}")
            print(f"short stock threshold {single_date['per_short'][0]}\n\
                date {single_date[single_date['pred_ls'] == -1].drop(['per_long', 'per_short'], axis = 1)}")
        
        ################################################################
        single_returns = single_date["return"]*single_date["pred_ls"]*single_date["weight"] # compute returns with  different weight
        rplsw = single_date
        market = pd.DataFrame([single_date["return"].mean()], columns = ["return"])
        performance = [single_returns.sum()]
        date_ = [date]
        performance = pd.DataFrame(performance, columns=["performance"])
        performance["ymd"] = date_
        market["ymd"] = date_
            
        return performance, rplsw, market # true, pred, long, short, weight
    """

    # ======================================================================
    # No tune for NN
    """
    def train_NN(self, inputs, labels, inputs_test, labels_test, lr, epoch):
        dataset = Data(torch.tensor(inputs, dtype = torch.float32).to(device),
                       torch.tensor(labels, dtype = torch.float32).to(device))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        # print(inputs.shape, labels.shape)
        criterion = nn.MSELoss()
        learning_rate = lr
        epoch = epoch
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_old = 100000
        for epoch in range(epoch):
            loss = 0
            for k, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.model(x)
                loss_i = criterion(outputs, y)
                loss_i.backward()
                optimizer.step()
                loss += loss_i
            if loss < loss_old:
                loss_old = loss
                torch.save(self.model.state_dict(), "simple_layer.pth")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr *1.2
                lr *= 1.2
            elif epoch == 0:
                torch.save(self.model.state_dict(), "simple_layer.pth")
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr *0.8
                lr *= 0.8
                self.model.load_state_dict(torch.load("simple_layer.pth"))
            if lr <= 1e-6:
                break
            # print(f"epoch {epoch} loss: {loss}")
        pred = self.model(torch.tensor(inputs_test, dtype = torch.float32).to(device))
        y_train_hat = self.model(torch.tensor(inputs, dtype = torch.float32).to(device))
        return pred.detach().cpu().numpy(), y_train_hat.cpu().detach().numpy(), loss
    """
    # ======================================================================