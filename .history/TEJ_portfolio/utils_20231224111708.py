import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence
import numpy as np
import matplotlib.pyplot as plt


def split_X_y(data):    
    # Split data
    try:
        data.set_index(["coid", "mdate"], inplace = True)
    except:
        pass
    finally:
        X = data.drop(["return", "Industry_Eng"], axis = 1)
        y = data["return"]
    return X, y

def fit_linear(X, y):
    # 
    linear_reg_ = sm.OLS(y, sm.add_constant(X))
    linear_reg = linear_reg_.fit()
    return linear_reg

def hat_linear(linear_reg, X):
    # predict
    hat = linear_reg.predict(sm.add_constant(X))
    return hat

def adj_rsqaure(n, k, rsquared):
    #
    return 1 - ((1 - rsquared) * (n - 1) / (n - k - 1))

def get_adj_close(data):
    # adj Close
    data["Adj Close"] = data["Close"] * data["Adjust_Factor"]
    return data

def data_to_week(data): 
    """
    data: daily data
    """
    # transform daily data to weekly
    data_week = data[data["mdate"].dt.weekday == 4]
    return data_week

def data_get_returns(data):
    # get data return: daily/weekly/monthly
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
    # integrate function for transform and clean data from daily to weekly data
    data_week = data_to_week(data)
    data_week_return = data_get_returns(data_week)
    return data_week_return

def fill_missing_value_dropna(data):
    # Function to fill missing values within each group
    def fillna_group(group):
        group = group.fillna(method = 'ffill').dropna()
        return group

    # Apply the function to each group using groupby
    df_filled = data.groupby('coid').apply(fillna_group)

    # Reset index after groupby to remove the multi-index
    df_filled = df_filled.droplevel(level = 0)
    return df_filled

class transform_preprocess:
    def __init__(self):
        pass
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
    def transform(X_train, X_test, scale_method = 'Standard', winsor = False):
        '''
        X_train
        X_test
        scale = 'Standard', 'MinMax'
        winsorise
        '''
        if scale_method == 'Standard':
            pass
        elif scale_method == 'MinMax':
            scaler = MinMaxScaler()
            scale_X_train = scaler.fit_transform(X_train)
            scale_train = pd.DataFrame(scale_X_train, columns = list(X_train.columns)).\
                set_index(X_train.index).\
                reset_index()

            scaler = MinMaxScaler()
            scale_X_test = scaler.fit_transform(X_test)
            scale_X_test = pd.DataFrame(scale_X_test, columns = list(X_test.columns)).\
                set_index(X_test.index).\
                reset_index()
        return scale_train, scale_X_test
    


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
    

def graphic_diagnostic(linear_reg):
    # Q-Q plot
    residuals = linear_reg.resid
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    '''
    # Scatter plot of x vs y
    axs[0, 0].scatter(X_train, y_train)
    axs[0, 0].set_title('Scatter Plot')
    '''
    # Histogram of Residuals
    hist, bin_edges = np.histogram(residuals, bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[0, 0].plot(bin_centers, hist, 'r-', linewidth=2, label='reseid freq', color='magenta')
    axs[0, 0].hist(residuals, bins=20, density=True, color='darkred') # , edgecolor='black',
    x_normal = np.linspace(min(residuals), max(residuals), 100)
    p_normal = stats.norm.pdf(x_normal, np.mean(residuals), np.std(residuals))
    axs[0, 0].plot(x_normal, p_normal, 'k', linewidth=2, label='normal')
    axs[0, 0].set_title('Histogram of Residuals with Normal Line')

    # Residuals vs Fitted values (hat from train X)
    axs[0, 1].scatter(linear_reg.fittedvalues, residuals, color='darkred')
    axs[0, 1].axhline(y=0, linestyle='--', color='magenta')
    axs[0, 1].set_title('Residuals vs Fitted')

    # Q-Q plot
    qqplot(residuals, line='s', ax=axs[1, 0], markerfacecolor='darkred')
    axs[1, 0].set_title('Q-Q Plot')

    # Influence Plot
    influence = OLSInfluence(linear_reg)
    (c, _) = influence.cooks_distance
    axs[1, 1].stem(np.arange(len(c)), c, markerfmt=",", linefmt="C0-", basefmt="C0-")
    axs[1, 1].set_title('Cook\'s Distance')

    plt.tight_layout()
    plt.show()

def remove_extreme_y(train, n=10):
    # Filter the DataFrame to keep only rows within the specified range
    """
    train: train data
    n: remove extreme high and low data of y
    """
    n = 10
    top_n_values = train.nlargest(n, 'return')
    bottom_n_values = train.nsmallest(n, 'return')
    train_filter = train[(train['return'] < top_n_values.min()['return']) & \
                        (train['return'] > bottom_n_values.max()['return'])]
    return train_filter












