import copy
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def xgboost(
        param,
        fnl_df,             
        tune = True):
        
    # data
    (X_train, y_train, X_test, y_test) = fnl_df

    if tune == True:
        print(f"tuning xgboost")
        grid_s = GridSearchCV(estimator=XGBRegressor(), param_grid=param, cv = 5)
        grid_s.fit(np.array(X_train), y_train)
        print("Finish Tuning")
        best_params = grid_s.best_params_
        tuned_model = XGBRegressor(**best_params)
        best_model = grid_s.best_estimator_
        with open(
            f"C:/Users/USER/Desktop/portfolio/temp_params_tuned_per_6month/xgboost.pickle",
            "wb") as f:
            pickle.dump(tuned_model, f)
    else:
        with open(
            f"C:/Users/USER/Desktop/portfolio/temp_params_tuned_per_6month/xgboost.pickle",
            "rb") as f:
            best_model = pickle.load(f)
            best_model.fit(np.array(X_train), y_train)
    test_hat = pd.DataFrame(best_model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
    train_hat = pd.DataFrame(best_model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
    loss = "na"
    return test_hat, train_hat, loss
