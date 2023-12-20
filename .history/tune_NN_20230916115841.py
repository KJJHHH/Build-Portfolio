
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
            