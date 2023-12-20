est)), columns = ["prediction"]).set_index(y_test.index)
         y_train_hat = pd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
         loss = "na"
     elif model_strategy["model"] == "random forest":
         grid_s = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param, cv = 5)
         grid_s.fit(np.array(X_train), y_train)
         best_model = grid_s.best_estimator_
         pred = pd.DataFrame(best_model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
         y_train_hat = best_model = grid_s.best_estimator_
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
        pred = self.model(torch.tensor(inputs_test, dtype = torch.float32).to(device))
        y_train_hat = self.model(torch.tensor(inputs, dtype = torch.float32).to(device))
        return pred.detach().cpu().numpy(), y_train_hat.cpu().detach().numpy(), loss

    def tune_NN(self, inputs):
        pass

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
            
        return performance, rplsw, market # true, pred, long, short, weightpd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
         loss = "na"
        elif model_strategy["model"] == "svm": # False: 
            grid_s = GridSearchCV(estimator=SVR(), param_grid=param, cv = 5)
            grid_s.fit(np.array(X_train), y_train)
            "model = SVR(**param)"
            "model.fit(np.array(X_train), y_train)"
            