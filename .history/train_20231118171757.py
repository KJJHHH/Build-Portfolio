from backtesting import Backtest, Strategy 
from backtesting.lib import crossover
from backtesting.test import SMA
from talib import abstract
import talib
import copy
import twstock
from datetime import datetime
import datetime as d
from pandas_datareader import data
import yfinance as y 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import gc
# ------------------------
from model import Data, Net

gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tuning_nn():
    pass

class summary():
    def __init__(self, dl_model = None):
        super(summary, self).__init__()
        if dl_model is not None:
            self.model = dl_model

    def traintest_period(self, # get start and end date for train and test and future predict
        train_start, train_end, test_end,  train_size = 5, test_size = 1, 
        begin = "begin", future_predicting = False): 
        if future_predicting == True:
            if begin == "begin":            
                train_start = pd.Timestamp(
                    year = pd.Timestamp.now().year-5, 
                    month = pd.Timestamp.now().month, 
                    day = 1) # , tz = "Asia/Taipei"
                train_end = train_start + pd.DateOffset(days=365*train_size)
                test_end = train_end + pd.DateOffset(months=test_size)
        else:
            if begin == "begin":            
                train_start = pd.Timestamp(year = train_end-5, month = 1, day = 1)# , tz = "Asia/Taipei"
                train_end = train_start + pd.DateOffset(days=365*train_size)
                test_end = train_end + pd.DateOffset(months=test_size)
            else: # begin == update
                train_start = train_start + pd.DateOffset(months=test_size)
                train_end = train_end + pd.DateOffset(months=test_size)
                test_end = test_end + pd.DateOffset(months=test_size)
        return train_start, train_end, test_end
    
    def Xy(self, data, train_start, train_end, test_end): # X_train, test, y_train, test
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
    
    def data_train(self, fnl_df, model_strategy, param, longshort_thres = None):        
        (X_train, y_train, X_test, y_test) = fnl_df
        data_tp = y_test    
        print(param)
        if model_strategy["model"] == "linear": # False: 
            model = LinearRegression(**param)
            model.fit(np.array(X_train), y_train)
            pred = pd.DataFrame(model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
            y_train_hat = pd.DataFrame(model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
            loss = "na"
        elif model_strategy["model"] == "decision tree": # False: 
            grid_s = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=param, cv = 5)
            grid_s.fit(np.array(X_train), y_train)
            print("Finish Tuning")
            "best_params = grid_s.best_params_"
            "grid_s = DecisionTreeRegressor(**param)"
            best_model = grid_s.best_estimator_
            pred = pd.DataFrame(best_model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
            y_train_hat = pd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
            loss = "na"
        elif model_strategy["model"] == "xgboost": # False: 
            grid_s = GridSearchCV(estimator=XGBRegressor(), param_grid=param, cv = 5)
            grid_s.fit(np.array(X_train), y_train)
            best_model = grid_s.best_estimator_
            pred = pd.DataFrame(best_model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
            y_train_hat = pd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
            loss = "na"
        elif model_strategy["model"] == "random forest":
            grid_s = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param, cv = 5)
            grid_s.fit(np.array(X_train), y_train)
            best_model = grid_s.best_estimator_
            pred = pd.DataFrame(best_model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
            y_train_hat = pd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
            loss = "na"
        elif model_strategy["model"] == "svm": # False: 
            grid_s = GridSearchCV(estimator=SVR(), param_grid=param, cv = 5)
            grid_s.fit(np.array(X_train), y_train)
            "model = SVR(**param)"
            "model.fit(np.array(X_train), y_train)"
            best_model = grid_s.best_estimator_
            pred = pd.DataFrame(best_model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
            y_train_hat = pd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
            loss = "na"
        elif model_strategy["model"] == "elastic net":
            grid_s = GridSearchCV(estimator=ElasticNet(), param_grid=param, cv = 5)
            grid_s.fit(np.array(X_train), y_train)
            best_model = grid_s.best_estimator_
            pred = pd.DataFrame(best_model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
            y_train_hat = pd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
            loss = "na"
        elif model_strategy["model"] == "neural network":
            X_train, y_train, X_test = \
                np.array(X_train), np.array(y_train), np.array(X_test) # y_test stay dataframe 
            pred, y_train_hat, loss = self.train_NN(X_train, y_train, X_test, y_test, **param)
            pred = pd.DataFrame(pred, columns=["prediction"]).set_index(y_test.index)
            'print(f"date {y_test.index.values}: loss{loss}")'

        data_tp = pd.concat([y_test, pred], axis = 1)  
        longshort_strategy = y_train_hat # pred
        if longshort_thres == None:
            longshort_thres = {}
            longshort_thres["long"] = np.percentile(longshort_strategy, model_strategy["long"]) 
            longshort_thres["short"] = np.percentile(longshort_strategy, model_strategy["short"]) 
            data_tp["per_long"] = np.percentile(longshort_strategy, model_strategy["long"]) 
            data_tp["per_short"] = np.percentile(longshort_strategy, model_strategy["short"]) 
        else:
            longshort_thres["long"] =  (longshort_thres["long"] + 
                                        np.percentile(longshort_strategy, model_strategy["long"]))/2
            longshort_thres["short"] = (longshort_thres["short"] +
                                        np.percentile(longshort_strategy, model_strategy["short"]))/2
            data_tp["per_long"] = longshort_thres["long"]
            data_tp["per_short"] = longshort_thres["short"]

        return data_tp, longshort_thres, loss
    
    def train_NN(self, inputs, labels, inputs_test, labels_test, lr, epoch):
        dataset = Data(torch.tensor(inputs, dtype = torch.float32).to(device),
                       torch.tensor(labels, dtype = torch.float32).to(device))
        train_loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=True)
        criterion = nn.MSELoss()
        learning_rate = lr
        epoch = epoch
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_old = 100000
        for epoch in range(epoch):
            loss = 0
            for k, (x, y) in  enumerate(train_loader):
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
            print(f"epoch{epoch}: loss)
        pred = self.model(torch.tensor(inputs_test, dtype = torch.float32).to(device))
        y_train_hat = self.model(torch.tensor(inputs, dtype = torch.float32).to(device))
        return pred.detach().cpu().numpy(), y_train_hat.cpu().detach().numpy(), loss

    def compute_ret(self, data_tp, not_short = False, print_detail = True): # class and returns
        data_tp = pd.DataFrame(data_tp)
        date = data_tp.reset_index()["ymd"].unique()

        date_ = []
        rplsw = pd.DataFrame() # true, pred, long, short, weight
        performance = []
        market = pd.DataFrame()
        for k in date: # No really need loop, just incase two month in same rolling 
            single_date = data_tp.reset_index()[data_tp.reset_index()["ymd"] == k].set_index(["code", "ymd"])
            single_date["pred_ls"] = np.where(
                single_date["prediction"] > single_date["per_long"], 1,
                np.where(single_date["prediction"] < single_date["per_short"], -1, 0))
            if not_short == True:
                single_date["pred_ls"] = np.where(single_date["prediction"] > single_date["per_long"], 1, 0)
            single_date['weight'] = abs(
                single_date["prediction"]/sum(abs(single_date["prediction"]*single_date["pred_ls"])))
            
            if print_detail == True:
                print(f"check weight {abs((single_date['weight']*single_date['pred_ls'])).sum()}")
                print(f"long stock threshold {single_date['per_long'][0]}\n\
                    date {single_date[single_date['pred_ls'] == 1].drop(['per_long', 'per_short'], axis = 1)}")
                print(f"short stock threshold {single_date['per_short'][0]}\n\
                    date {single_date[single_date['pred_ls'] == -1].drop(['per_long', 'per_short'], axis = 1)}")
            single_returns = single_date["return"]*single_date["pred_ls"]*single_date["weight"] # compute returns with  different weight
            rplsw = pd.concat([rplsw, single_date], axis = 0)
            market = pd.concat([market, pd.DataFrame([single_date["return"].mean()], columns = ["return"])], axis = 0)
            performance.append(single_returns.sum())
            date_.append(k)
        performance = pd.DataFrame(performance, columns=["performance"])
        performance["ymd"] = date_
        market["ymd"] = date_
            
        return performance, rplsw, market # true, pred, long, short, weight