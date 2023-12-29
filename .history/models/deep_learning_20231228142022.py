import copy
import pickle
import pandas as pd
import numpy as np
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
from models.dl_model import Data, Net_tune, Net_tuned

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class deep_learning():
    def __init__(self, industry, param, n_trials, input_size):
        super(deep_learning, self).__init__()
        if input_size is not None:
            self.input_size = input_size
            self.dir = os.getcwd()
            self.industry = industry
            self.param = param
            self.config = None
            self.n_trials = n_trials

    def dl_tuning_pipeline(self, data, n_trials):
        """
        data: (X_train, y_train, X_test, y_test)
        n_trials: if do not tune -> set to 1
        """
        
        (X_train, y_train, X_test, y_test) = data
        X_train = torch.tensor(np.array(X_train), dtype = torch.float32).to(device)
        y_train = torch.tensor(np.array(y_train), dtype = torch.float32).to(device)
        X_test = torch.tensor(np.array(X_test), dtype = torch.float32).to(device)

        #######################################
        # tuning
        try:
            print("===> tune start")
            trial = self.tune(X_train, y_train)        
            self.config = trial.params              
            print(self.config)
        except:
            print("===> Error tuning, skip tune")

        # train
        print("===> training start")
        data_tensor = \
            (
            X_train, 
            y_train, 
            X_test, 
            torch.tensor(np.array(y_test), dtype = torch.float32).to(device)
            )
        test_hat, train_hat, test_loss = self.tune_train(data_tensor)
        test_hat = pd.DataFrame(test_hat, columns=["prediction"]).set_index(y_test.index)
        return test_hat, train_hat, self.config


    #######################################################################
    # Tune NN

    # Functions for tune
    def get_mnist(self):
        with open("models/temp_data/temp_X", "rb") as f:
            X_train = pickle.load(f)
        with open("models/temp_data/temp_y", "rb") as f:
            y_train = pickle.load(f)
        
        batch_size = self.param["batch_size"]
        data = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = data[0], data[1], data[2], data[3]
        dataset = Data(X_train,y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        val_dataset = Data(X_val,y_val)
        val_loader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)

        return train_loader, X_val, y_val

    def objective(self, trial: optuna.trial.Trial):
        # Generate the model. 
        model = Net_tune(trial, input_size=89).to(device)

        # Generate the optimizers.
        batch_size = self.param["batch_size"]
        n_train_examples = batch_size * 60
        n_val_examples = batch_size * 10
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
                if batch_idx * batch_size >= n_train_examples:
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
                    model.load_state_dict(torch.load('models/temp_data/model.pth'))
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
                except:
                    torch.save(model.state_dict(), 'models/temp_data/model.pth')
                    loss_old = loss_all
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
                    print(">_<load model failed")
            if optimizer.param_groups[0]['lr'] < 1e-5:
                break
            else:
                torch.save(model.state_dict(), 'models/temp_data/model.pth')
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

    def tune(self, X_train, y_train):
        with open("models/temp_data/temp_X", "wb") as f:
            pickle.dump(X_train, f)
        with open("models/temp_data/temp_y", "wb") as f:
            pickle.dump(y_train, f)
        while True:
            try:
                study = optuna.create_study(direction="maximize")
                study.optimize(self.objective, n_trials=self., timeout=600000)
                pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
                complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
                trial = study.best_trial
                break
            except:
                pass
        return trial

    def tune_train(self, data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
        batch_size = self.param["batch_size"]
        dataset = Data(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        model = Net_tuned(self.config, layers = self.config["n_layers"], input_size = 89).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"])

        loss_f = nn.MSELoss()
        loss_old = 10000000
        loss_all = 0
        
        #/////////////////////////////////////////////////////////
        # training
        for s in range(self.config["epochs"]):
            loss_all = 0
            for i, (X, y) in enumerate(train_loader):
                output = torch.squeeze(model(X.to(device)))
                loss = loss_f(output, y.to(device))
                loss += self.config["reg_coef"] \
                    * torch.mean(torch.cat([param.view(-1)**2 for param in model.parameters()]))
                loss.backward()

                if self.config["scheduler"] == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0, last_epoch=- 1, verbose=False)
                    scheduler.step()

                if self.config["scheduler"] == "None":
                    optimizer.step()

                optimizer.zero_grad()
                loss_all += loss
            if loss_all >= loss_old:
                try:
                    # print(" >_< Not training in this epcoh")
                    model.load_state_dict(torch.load('models/temp_data/model.pth'))
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
                torch.save(model.state_dict(), 'models/temp_data/model.pth')
                loss_old = loss_all
                # print(f"Training epoch:{s} | training loss = {loss_all/i+1}")

            # print(f"training output: {output[:16]}")
        # torch.save(model.state_dict(), 'models/temp_data/model.pth')

        #/////////////////////////////////////////////////////////
        # test
        with torch.no_grad():
            output = torch.squeeze(model(X_test.to(device)))
            test_loss = loss_f(output, y_test.to(device))
            y_train_hat = torch.squeeze(model(X_train.to(device)))
        
        return output, y_train_hat, test_loss

    #######################################################################