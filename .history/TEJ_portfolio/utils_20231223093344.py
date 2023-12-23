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

# adj Close
def get_adj_close(data):
    data["Adj Close"] = data["Close"] * data["Adjust_Factor"]
    return data

# transform daily data t oweekly
def data_to_week(data): # daily data
    data_week = data[data["mdate"].dt.weekday == 4]
    return data_week

def data_get_returns(data):

    prc_return = "Adj Close"
    data_return = pd.DataFrame()

    for k, symbol in enumerate(data["coid"].unique()):
        print(f"{k} symbol: {symbol}")
        data_coid = data[data["coid"] == symbol].copy()
        data_coid["return"] = \
            (data_coid[prc_return].shift(-1) - data_coid[prc_return])/data_coid[prc_return]
        data_return = pd.concat([data_return, data_coid], axis = 0)
        
    return data_return

def date_to_week_pipeline(data):
    data_week = data_to_week(data)
    data_week_return = data_get_returns(data_week)
    return data_week_return