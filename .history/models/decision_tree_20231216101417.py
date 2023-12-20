import copy
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor 

def decision_tree(
        param,
        fnl_df,             
        tune = True):
    
    # data
    (X_train, y_train, X_test, y_test) = fnl_df
    if tune == True:
                print("tuning dt")
                grid_s = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=self.param, cv = 5)
                grid_s.fit(np.array(X_train), y_train)
                print("Finish Tuning")
                best_params = grid_s.best_params_
                tuned_model = DecisionTreeRegressor(**best_params)
                best_model = grid_s.best_estimator_
                with open(
                    f"C:/Users/USER/Desktop/portfolio/tunedparams_tempper6month/{model_strategy['model']}.pickle",
                    "wb") as f:
                    pickle.dump(tuned_model, f)
            else:
                with open(
                    f"C:/Users/USER/Desktop/portfolio/tunedparams_tempper6month/{model_strategy['model']}.pickle",
                    "rb") as f:
                    best_model = pickle.load(f)
                    best_model.fit(np.array(X_train), y_train)
            pred = pd.DataFrame(best_model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
            y_train_hat = pd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
            loss = "na"