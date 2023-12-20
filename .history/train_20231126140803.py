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
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import gc
import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# ------------------------
from model import Data, Net_tune, Net_tuned

gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class summary():
    def __init__(self, input_size = None):
        super(summary, self).__init__()
        if input_size is not None:
            self.input_size = input_size
            self.batch_size = 25
            self.dir = os.getcwd()
            self.n_train_examples = self.batch_size * 30
            self.n_val_examples = self.batch_size * 10

    def traintest_period(self, # get start and end date for train and test and future predict
        train_start, train_end, test_end,  train_size = 5, test_size = 1, 
        begin = "begin", future_predicting = False): 
        if future_predicting == True:
            if begin == "begin":            
                train_start = pd.Timestamp(
                    year = pd.Timestamp.now().year-train_size, 
                    month = pd.Timestamp.now().month, 
                    day = 1) # , tz = "Asia/Taipei"
                train_end = train_start + pd.DateOffset(days=365*train_size)
                test_end = train_end + pd.DateOffset(months=test_size)
        else:
            if begin == "begin":            
                train_start = pd.Timestamp(year = train_end-train_size, month = test_size, day = 1)# , tz = "Asia/Taipei"
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
    
    def data_train(self, fnl_df, model_strategy, param, longshort_thres = None, tune = False, ensemble_addmodel = False):        
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
            if tune == True:
                print(f"tuning {model_strategy['model']}")
                grid_s = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=param, cv = 5)
                grid_s.fit(np.array(X_train), y_train)
                print("Finish Tuning")
                best_params = grid_s.best_params_
                tuned_model = DecisionTreeRegressor(best_params)
                best_model = grid_s.best_estimator_
                with open(f"/tunedparams_tempper6month/{model_strategy['model']}.pickle", "wb") as f:
                    pickle.dump(tuned_model)
            else:
                with open(f"/tunedparams_tempper6month/{model_strategy['model']}.pickle", "rb") as f:
                    best_model = pickle.load(f)
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
            #######################################
            # tuning
            print("===> tune start")
            n_trial = model_strategy["n_trials"]
            trial = self.tune(torch.tensor(np.array(X_train), dtype = torch.float32).to(device),
                torch.tensor(np.array(y_train), dtype = torch.float32).to(device), n_trial)
            config = trial.params
            print(config)
            
            # store config
            config_ = [config]
            try:
                with open(f"nn_config/nnconfig_{n_trial}trial.pickle", "rb") as f: 
                    config_past = pickle.load(f)
                    config_ = config_past + config_
            except:
                pass
            with open(f"nn_config/nnconfig_{n_trial}trial.pickle", "wb") as f: 
                pickle.dump(config_, f)

            # train
            print("===> training start")
            data = (torch.tensor(np.array(X_train), dtype = torch.float32).to(device),
                torch.tensor(np.array(y_train), dtype = torch.float32).to(device),
                torch.tensor(np.array(X_test), dtype = torch.float32).to(device),
                torch.tensor(np.array(y_test), dtype = torch.float32).to(device))
            y_hat, y_train_hat, loss = self.tune_train(config, data)
            #######################################
            #######################################
            # No tune
            # pred, y_train_hat, loss = self.train_NN(X_train, y_train, X_test, y_test, **param)
            #######################################
            pred = pd.DataFrame(y_hat, columns=["prediction"]).set_index(y_test.index)
            'print(f"date {y_test.index.values}: loss{loss}")'

        data_tp = pd.concat([y_test, pred], axis = 1)  
        longshort_strategy = y_train_hat # pred
        if longshort_thres == None:
            longshort_thres = {}
            longshort_thres["long"] = np.percentile(longshort_strategy, model_strategy["long"]) 
            longshort_thres["short"] = np.percentile(longshort_strategy, model_strategy["short"]) 
            data_tp["per_long"] = np.percentile(longshort_strategy, model_strategy["long"]) 
            data_tp["per_short"] = np.percentile(longshort_strategy, model_strategy["short"]) 
        if model_strategy["model"] == "ensemble" and ensemble_addmodel == True:
            return data_tp, longshort_thres, loss
        else:
            longshort_thres["long"] =  (longshort_thres["long"] + 
                                        np.percentile(longshort_strategy, model_strategy["long"]))/2
            longshort_thres["short"] = (longshort_thres["short"] +
                                        np.percentile(longshort_strategy, model_strategy["short"]))/2
            data_tp["per_long"] = longshort_thres["long"]
            data_tp["per_short"] = longshort_thres["short"]

        return data_tp, longshort_thres, loss # test_loss
    
    # ======================================================================
    # No tune 
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

    # ======================================================================
    # Functions for tune
    def get_mnist(self):
        with open("temp_X", "rb") as f:
            X_train = pickle.load(f)
        with open("temp_y", "rb") as f:
            y_train = pickle.load(f)

        data = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = data[0], data[1], data[2], data[3]
        dataset = Data(X_train,y_train)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        val_dataset = Data(X_val,y_val)
        val_loader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)

        return train_loader, X_val, y_val

    def objective(self, trial: optuna.trial.Trial):
        # Generate the model. 
        model = Net_tune(trial, input_size=89).to(device)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # X_train, X_val, y_train, y_val
        train_loader, X_val, y_val = self.get_mnist()

        # Training of the model.
        epochs = trial.suggest_int("epochs", 150, 300)
        loss_old = 10000000000
        for epoch in range(epochs):
            model.train()
            loss_all = 0
            for batch_idx, (X, y) in enumerate(train_loader):
                if batch_idx * self.batch_size >= self.n_train_examples:
                    break
                pred = model(torch.tensor(X, dtype = torch.float32))

                loss = F.mse_loss(pred, torch.tensor(y, dtype = torch.float32))
                loss += trial.suggest_float("reg_coef", 1e-5, 1e-1, log=True) \
                * torch.mean(torch.cat([param.view(-1)**2 for param in model.parameters()]))
                loss.backward()
                sche = trial.suggest_categorical("scheduler", ["None"])
                if sche == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0, last_epoch=- 1, verbose=False)
                    scheduler.step()

                if sche == "None":
                    optimizer.step()

                optimizer.zero_grad()
                loss_all += loss
            if loss_all > loss_old:
                try:
                    model.load_state_dict(torch.load('model.pth'))
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
                except:
                    torch.save(model.state_dict(), 'model.pth')
                    loss_old = loss_all
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
                    print(">_<load model failed")
            if optimizer.param_groups[0]['lr'] < 1e-5:
                break
            else:
                torch.save(model.state_dict(), 'model.pth')
                loss_old = loss_all
            # print(f"Tuning epoch {epoch} with loss | {loss_all}")
        # eval
        model.eval()
        loss = 0
        with torch.no_grad():
            pred = model(X_val)
            if pred[0] == pred[1] or pred[1] == pred[2]:
                print("bad model")
                loss = 1000000000
            loss += F.mse_loss(pred, y_val)

        loss_mean = -loss / (batch_idx+1)

        trial.report(loss_mean, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return loss_mean

    def tune(self, X_train, y_train, n_try = 10):
        with open("temp_X", "wb") as f:
            pickle.dump(X_train, f)
        with open("temp_y", "wb") as f:
            pickle.dump(y_train, f)
        while True:
            try:
                study = optuna.create_study(direction="maximize")
                study.optimize(self.objective, n_trials=n_try, timeout=600000)

                pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
                complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

                trial = study.best_trial
                break
            except:
                pass
        return trial

    def tune_train(self, config, data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
        dataset = Data(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        model = Net_tuned(config, layers = config["n_layers"], input_size = 89).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        loss_f = nn.MSELoss()
        loss_old = 10000000
        loss_all = 0
        for s in range(config["epochs"]):# config["epochs"]
            loss_all = 0
            for i, (X, y) in enumerate(train_loader):
                output = torch.squeeze(model(X.to(device)))
                loss = loss_f(output, y.to(device))
                loss += config["reg_coef"] \
                    * torch.mean(torch.cat([param.view(-1)**2 for param in model.parameters()]))
                loss.backward()

                if config["scheduler"] == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0, last_epoch=- 1, verbose=False)
                    scheduler.step()

                if config["scheduler"] == "None":
                    optimizer.step()

                optimizer.zero_grad()
                loss_all += loss
            if loss_all >= loss_old:
                try:
                    # print(" >_< Not training in this epcoh")
                    model.load_state_dict(torch.load('model.pth'))
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
                except:
                    torch.save(model.state_dict(), 'model.pth')
                    loss_old = loss_all
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
                    print(" >_< load model failed")
                    # print(f"epoch:{s} | training loss = {loss_all/i+1}")
                if optimizer.param_groups[0]['lr'] < 1e-5:
                    # print("lr too small")
                    break
            else:
                torch.save(model.state_dict(), 'model.pth')
                loss_old = loss_all
                # print(f"Training epoch:{s} | training loss = {loss_all/i+1}")

            # print(f"training output: {output[:16]}")

        with torch.no_grad():
            output = torch.squeeze(model(X_test.to(device)))
            test_loss = loss_f(output, y_test.to(device))
            y_train_hat = torch.squeeze(model(X_train.to(device)))
        return output, y_train_hat, test_loss
    # ======================================================================

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
            print(f"Check sum of weight: {(single_date['weight']*single_date['pred_ls']).sum()}")
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
    