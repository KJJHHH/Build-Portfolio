import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

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

# predict
def predict(linear_reg, X):
    hat = linear_reg.predict(sm.add_constant(X))
    return hat

#
def adj_rsqaure(n, k, rsquared):
    return 1 - ((1 - rsquared) * (n - 1) / (n - k - 1))

# adj Close
def get_adj_close(data):
    data["Adj Close"] = data["Close"] * data["Adjust_Factor"]
    return data

# transform daily data to weekly
def data_to_week(data): # daily data
    data_week = data[data["mdate"].dt.weekday == 4]
    return data_week

# get data return: daily/weekly/monthly
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

# integrate function for transform and clean data from daily to weekly data
def date_to_week_pipeline(data):
    data_week = data_to_week(data)
    data_week_return = data_get_returns(data_week)
    return data_week_return

# Create a sample DataFrame

# Function to fill missing values within each group
def fill_missing_value_dropna(data):

    def fillna_group(group):
        group = group.fillna(method = 'ffill').dropna()
        return group

    # Apply the function to each group using groupby
    df_filled = data.groupby('coid').apply(fillna_group)

    # Reset index after groupby to remove the multi-index
    df_filled = df_filled.droplevel(level = 0)
    return df_filled

def standardise_winsorise_by_date(data):
    data_scaler_winsorise = pd.DataFrame()

    def standardise(data):
        index = data.index
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns).\
                                    set_index(index)  
        return data
    
    def winsorise(data):
        for col in data.columns:
            data[col] = winsorize(data[col], limits=[0.001, 0.001], inplace=False)             
        return data
    
    data.set_index(["coid", "mdate", "Industry_Eng", "Close", "Open", "return"], inplace=True)
    data = data.groupby(['mdate']).apply(standardise)
    data = data.groupby(['mdate']).apply(winsorise)
    
    data = data.reset_index()
    return data