import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState

from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from torchvision import datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 




class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])  

class Net_tune(nn.Module):
    def __init__(self, trial, input_size = 89):
        super(Net_tune, self).__init__()
        self.num_layers = trial.suggest_int("n_layers", 1, 10)
        self.hidden_layers = nn.ModuleList()
        self.activation = trial.suggest_categorical("active", [True, False])
        self.norm = nn.BatchNorm1d(1)
        for i in range(self.num_layers):
            output_size = trial.suggest_int("hidden_nodes{}".format(i), 4, 512)
            self.hidden_layers.append(nn.Linear(input_size, output_size).to(device))
            if self.activation == True:
              self.hidden_layers.append(nn.ReLU().to(device))
            self.hidden_layers.append(nn.BatchNorm1d(output_size).to(device))
            input_size = output_size

        self.output_layer = nn.Linear(output_size, 1).to(device)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        # output = self.norm(output)
        return output
    
class Net_tuned(nn.Module):
    def __init__(self, config, layers, input_size = 89):
        super(Net_tuned, self).__init__()
        self.num_layers = layers
        self.norm = nn.BatchNorm1d(1)
        self.hidden_layers = nn.ModuleList()
        for i in range(self.num_layers):
            output_size = config[f"hidden_nodes{i}"]
            self.hidden_layers.append(nn.Linear(input_size, output_size).to(device))
            if config["active"] == True:
              self.hidden_layers.append(nn.ReLU().to(device))
            self.hidden_layers.append(nn.BatchNorm1d(output_size).to(device))
            input_size = output_size

        self.output_layer = nn.Linear(output_size, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x).to(device)
        output = self.output_layer(x).to(device)
        # output = self.norm(output)
        return output
    
