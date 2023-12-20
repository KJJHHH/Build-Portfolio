import torch.nn as nn

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


# ======================================================================
# functions for train

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