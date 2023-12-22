import statsmodels.api as sm
import pandas as pd

# Split data
def split_X_y(data):
    try:
        data.set_index(["coid", "mdate"], inplace = True)
    except:
        pass
    finally:
        X = data.drop(["return", "Industry_Eng"], axis = 1)
        y = data["return"]
    return X, y

# 
def fit_linear(X, y):
    linear_reg_ = sm.OLS(y, sm.add_constant(X))
    linear_reg = linear_reg_.fit()

    return linear_reg