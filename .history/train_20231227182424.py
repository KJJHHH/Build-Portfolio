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
    Build the pipeline of data and train in SINGLE period (month),
    which include functions:
    0. Initialise:
        train and test date and data
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
    def __init__(self, param, input_size = None):
        super(summary, self).__init__()
        if input_size is not None:
            self.param = param
            self.input_size = input_size
            self.dir = os.getcwd()

    def initialise(train_end):
        train_start = pd.Timestamp(year = train_end-train_size, month = test_size, day = 1)# , tz = "Asia/Taipei"
        test_start = train_start + pd.DateOffset(days=365*train_size)
        test_end = test_start + pd.DateOffset(months=test_size)
        
    # get start and end date for train and test and future predict
    def traintest_update(self, 
        train_start, test_start, test_end,  train_size = 5, test_size = 1, 
        begin = "begin"): 
        if begin == "begin":            
            train_start = pd.Timestamp(year = test_start-train_size, month = test_size, day = 1)# , tz = "Asia/Taipei"
            test_start = train_start + pd.DateOffset(days=365*train_size)
            test_end = test_start + pd.DateOffset(months=test_size)
        else: # begin == update
            train_start = train_start + pd.DateOffset(months=test_size)
            test_start = test_start + pd.DateOffset(months=test_size)
            test_end = test_end + pd.DateOffset(months=test_size)
        return train_start, test_start, test_end
    
    # X_train, test, y_train, test
    def Xy(self, data, train_start, train_end, test_end): 
        temp_d = copy.copy(data)
        d_train = temp_d.reset_index()[
            (temp_d.reset_index()["ymd"] >= train_start) & \
            (temp_d.reset_index()["ymd"] <= train_end)]\
            .set_index(["code", "ymd"])
        d_test = temp_d.reset_index()[
            (temp_d.reset_index()["ymd"] > train_end) & \
            (temp_d.reset_index()["ymd"] <= test_end)]\
            .set_index(["code", "ymd"])
        X_train = d_train.drop(["return"], axis = 1)
        y_train = d_train["return"]
        X_test = d_test.drop(["return"], axis = 1)
        y_test = d_test["return"]
        return X_train, y_train, X_test, y_test
    
    # train
    def train_lsdecision(
            self, 
            data,                  # data
            model_strategy,          # dict
            longshort_thres = None,  # dict
            tune = True
            ):        
        ############################################
        # For 1 single peoriod, 
        # train and decide the long and short boundary
        ############################################
        (X_train, y_train, X_test, y_test) = data
        data_tp = y_test    
        
        if model_strategy["model"] == "linear": # False: 
            test_hat, train_hat, loss = linear_regression(self.param, data, tune)
        elif model_strategy["model"] == "decision tree": # False: 
            test_hat, train_hat, loss = decision_tree(self.param, data, tune)
        elif model_strategy["model"] == "xgboost": # False: 
            test_hat, train_hat, loss = xgboost(self.param, data, tune)
        elif model_strategy["model"] == "random forest":
            test_hat, train_hat, loss = random_forest(self.param, data, tune)
        elif model_strategy["model"] == "svm": # False:
            test_hat, train_hat, loss = svm(self.param, data, tune)
        elif model_strategy["model"] == "elastic net":
            test_hat, train_hat, loss = elastic_net(self.param, data, tune)
        elif model_strategy["model"] == "neural network":
            n_trials = model_strategy["n_trials"]
            industry = model_strategy["industry"]
            dl = deep_learning(industry, self.param, self.input_size)
            test_hat, train_hat, loss = dl.dl_tuning_pipeline(data, n_trials, tune)


        data_tp = pd.concat([y_test, test_hat], axis = 1)  
        if model_strategy["ls_decision"][0] == "train":
            ls_decision = train_hat 
        else:
            ls_decision = test_hat

        # ====================================================
        # For first rolling prediction for all models and 
        # all rolling predictions for ensemble 
        # do not use running mean
        if (longshort_thres == None) or (model_strategy["ls_decision"][1] != "running"):   
            longshort_thres = {}
            longshort_thres["long"] = np.percentile(ls_decision, model_strategy["long"]) 
            longshort_thres["short"] = np.percentile(ls_decision, model_strategy["short"]) 
            data_tp["per_long"] = np.percentile(ls_decision, model_strategy["long"]) 
            data_tp["per_short"] = np.percentile(ls_decision, model_strategy["short"]) 
            return data_tp, longshort_thres, loss
        # Running mean for the thresholds
        else: 
            longshort_thres["long"] =  (longshort_thres["long"] + 
                                        np.percentile(ls_decision, model_strategy["long"]))/2
            longshort_thres["short"] = (longshort_thres["short"] +
                                        np.percentile(ls_decision, model_strategy["short"]))/2
            data_tp["per_long"] = longshort_thres["long"]
            data_tp["per_short"] = longshort_thres["short"]
            return data_tp, longshort_thres, loss # test_loss
        
        # ====================================================

    # 
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