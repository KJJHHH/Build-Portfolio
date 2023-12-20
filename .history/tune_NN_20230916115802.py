est)), columns = ["prediction"]).set_index(y_test.index)
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
            