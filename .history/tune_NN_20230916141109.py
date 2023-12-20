import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = 256
dir = os.getcwd()
n_train_examples = batch * 30
n_val_examples = batch * 10

# ======================================================================
# functions for tune
class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class Net_tune(nn.Module):
    def __init__(self, trial, input_size = 85):
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

def get_mnist():
    with open("temp_X", "rb") as f:
        X_train = pickle.load(f)
    with open("temp_y", "rb") as f:
        y_train = pickle.load(f)
    data = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = data[0], data[1], data[2], data[3]
    dataset = Data(X_train,y_train)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=True)
    val_dataset = Data(X_val,y_val)
    val_loader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)

    return train_loader, X_val, y_val

def objective(trial: optuna.trial.Trial):
    # Generate the model.
    model = Net_tune(trial).to(device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # X_train, X_val, y_train, y_val
    train_loader, X_val, y_val = get_mnist()

    # Training of the model.
    epochs = trial.suggest_int("epochs".format(i), 150, 300)
    loss_old = 10000000000
    for epoch in range(epochs):
        model.train()
        loss_all = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            if batch_idx * batch >= n_train_examples:
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
            print("load model failed")
          if optimizer.param_groups[0]['lr'] < 1e-5:
              break
        else:
            torch.save(model.state_dict(), 'model.pth')
            loss_old = loss_all
        print(f"epoch {epoch} | {loss_all}")
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

def tune(X_train, y_train, n_try = 1):
    with open("temp_X", "wb") as f:
        pickle.dump(X_train, f)
    with open("temp_y", "wb") as f:
        pickle.dump(y_train, f)
    if __name__ == "__main__":
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_try, timeout=600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        trial = study.best_trial
    return trial
# ======================================================================

# ======================================================================
# functions for train
class NetTune(nn.Module):
    def init(self, input_size, trial, hidden_size, output_size):
        super(NetTune, self).__init__()
        self.num_layers = trial.suggest_int("n_layers", 1, 10)
        self.layer_dims = [input_size]\
                            + [trial.suggest_int("hidden_nodes{}".format(i), 4, 512) for i in self.num_layers - 1]\
                            + [output_size]
        self.fcs = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i+1]) for i in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(self.layer_dims[i+1]) for i in range(self.num_layers)
        ])

    def forward(self, x):
        # x = self.norm0(x)
        for i, (fc, norm) in enumerate(zip(self.fcs, self.norms)):
            if i + 1 != self.num_layers:
                x = fc(x)
                x = norm(x)
                x = self.relu(x)
            else:
                out = fc(x)
        return out



def train(config, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
    dataset = Data(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=True)
    model = Net(config, layers = config["n_layers"], input_size = 85).to(device)
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
            print(" >_< Not training in this epcoh")
            model.load_state_dict(torch.load('model.pth'))
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
          except:
            torch.save(model.state_dict(), 'model.pth')
            loss_old = loss_all
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
            print(" >_< load model failed")
            print(f"epoch:{s} | training loss = {loss_all/i+1}")
          if optimizer.param_groups[0]['lr'] < 1e-5:
              print("lr too small")
              break
        else:
            torch.save(model.state_dict(), 'model.pth')
            loss_old = loss_all
            print(f"epoch:{s} | training loss = {loss_all/i+1}")

        print(f"training output: {output[:16]}")

    with torch.no_grad():
        output = torch.squeeze(model(X_test.to(device)))
        test_loss = loss_f(output, y_test.to(device))
    return output
# ======================================================================