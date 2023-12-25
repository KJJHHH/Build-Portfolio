import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr

class feature_engineer:
    def __init__(self):
        pass
    
    def to_final_index(self, data, final_index):
        data = data.reset_index().set_index(['coid', 'mdate'])
        return data.reindex(final_index)

    def transform_standard_winsor(self, X_train, X_test = None, scale_method = None, winsor = None):
        '''
        X_train
        X_test = None, X_test
        scale = None, 'Standard', 'MinMax'
        winsorise = True, False
        # after all process: index = coid, mdate
        '''
        
        def standard(data):
            index = data.index
            scaler = StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns).\
                                        set_index(index)  
            return data
    
        def minmax(data):
            index = data.index
            scaler = MinMaxScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns).\
                set_index(index)
            return data
        
        def winsorise(data):
            for col in data.columns:
                data[col] = winsorize(data[col], limits=[0.001, 0.001], inplace=False)             
            return data
        

        train_final_index = X_train.index
        test_final_index = X_test.index
        # scaling   
        if self.scale_method != None:
            X_train = X_train.reset_index().set_index('coid')
            if type(X_test) == pd.core.frame.DataFrame:
                X_test = X_test.reset_index().set_index('coid')

        if self.scale_method == 'Standard':
            X_train = X_train.groupby(['mdate']).apply(standard)
            X_train = self.to_final_index(X_train, train_final_index)
            if type(X_test) == pd.core.frame.DataFrame:
                X_test = X_test.groupby(['mdate']).apply(standard)
                X_test = self.to_final_index(X_test, test_final_index)
        elif self.scale_method == 'MinMax':
            X_train = X_train.groupby(['mdate']).apply(minmax)
            X_train = self.to_final_index(X_train, train_final_index)
            if type(X_test) == pd.core.frame.DataFrame:
                X_test = X_test.groupby(['mdate']).apply(minmax)
                X_test = self.to_final_index(X_test, test_final_index)

        # winsor
        if self.winsor == True:
            X_train = X_train.reset_index().set_index('coid')
            if type(X_test) == pd.core.frame.DataFrame:
                X_test = X_test.reset_index().set_index('coid')

            X_train = X_train.groupby(['mdate']).apply(winsorise)
            X_train = self.to_final_index(X_train, train_final_index)
            if type(X_test) == pd.core.frame.DataFrame:
                X_test = X_test.groupby(['mdate']).apply(winsorise)
                X_test = self.to_final_index(X_test, test_final_index)

        if type(X_test) == pd.core.frame.DataFrame:
            return X_train, X_test
        else:
            return X_train
    
    def transform_poly_selected_pearsonr(X, y):
        # 1. poly features
        # 2. Selects the features with a correlation coefficient of 0.005 or higher.
        """
        X: scaled data => reduced computation with 100000000*2
        """
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        selected_col = []
        print(X_poly.shape[1])
        for i in range(X_poly.shape[1]):
            print(i)
            corr, _ = pearsonr(X_poly[:, i], y)
            if corr >= 0.005:
                selected_col.append(i)

        X_poly = X_poly[:, selected_col]
        print(selected_col)
        return X_poly
    
    # PCA
    def transform_pca(X, var=10):
        # X max corr in corr matrix < 0.2? if not: do pca
        """
        X: X_poly selected with corr
        var: variables number after pca 
        """
        corr_matrix = np.corrcoef(X.T)
        np.fill_diagonal(corr_matrix, 0)
        max_corr = np.max(corr_matrix)
        print(f'check max corr: {max_corr}')
        if max_corr < 0.2:
            print('good, no pca')
            return X
        
        while True:
            step = 5
            pca = PCA(n_components=var)
            X_pca = pca.fit_transform(X)
            # check corr
            corr_matrix = np.corrcoef(X.T)
            np.fill_diagonal(corr_matrix, 0)
            max_corr = np.max(corr_matrix)
            print(f'check max corr: {max_corr}')
            if max_corr < 0.2:
                print('good')
                return X_pca
            else:
                var -= step


    def selected_bic(X):
        # selected variable with bic
        """
        X: X_pca, no multi collinearity
        """
        best_bic = 200
        selected_col = []

        while True:
            get_better = False
            for i in range(X.shape[1]):
                if i in selected_col:
                    continue

                linear_reg = fit_linear(X[:, selected_col + [i]], y_train)
                if linear_reg.bic < best_bic:
                    get_better = True
                    best_bic = linear_reg.bic
                    best_col = i
                    
            if get_better == False:
                break
            selected_col.append(best_col)
        X_selected_bic = X[:, selected_col]
        return X_selected_bic, selected_col
    


def update_data(data_week_return, start_train_date, train_size = 6):
    start_train_date = start_train_date + pd.DateOffset(weeks = 1)
    end_train_date = start_train_date + pd.DateOffset(weeks = train_size)
    start_test_date = end_train_date + pd.DateOffset(weeks = 1)
    end_test_date = start_test_date + pd.DateOffset(weeks = 1)
    train = data_week_return[(data_week_return["mdate"] < end_train_date) & \
                            (data_week_return["mdate"] >= start_train_date)]
    test = data_week_return[(data_week_return["mdate"] < end_test_date) & \
                            (data_week_return["mdate"] >= start_test_date)]
    return train, test, start_train_date, start_test_date

def split_X_y(data):    
    # Split data
    try:
        data = data.set_index(["coid", "mdate"]).copy()
    except:
        print('warning: no columns coid and mdate')
    finally:
        X = data.drop(["return", "Industry_Eng"], axis = 1).copy()
        y = data["return"].copy()
    return X, y

def fit_linear(X, y, HC = 'nonrobust'):
    # Don't add constant here!!
    # print(HC)
    linear_reg_ = sm.OLS(y, X, cov_type=HC)
    linear_reg = linear_reg_.fit()

    return linear_reg

def build_wls(X, y):    
    print("building wls")
    # fit linear 
    linear_reg = fit_linear(X, y)

    # Save the absolute values of the residuals of the OLS model
    y_resid = [abs(resid) for resid in linear_reg.resid]

    # Add constant according to statsmodels documentation to the fitted values of the OLS model
    X_resid = sm.add_constant(linear_reg.fittedvalues)

    # Create OLS model, fit, and print results
    mod_resid = sm.OLS(y_resid, X_resid)
    res_resid = mod_resid.fit()

    # Estimate of std. dev. (sigma)
    mod_fv = res_resid.fittedvalues

    # Calculate weights
    weights = 1 / (mod_fv**2)

    # fit WLS
    linear_reg_weighted = sm.WLS(y, X, weights = weights).fit()

    return linear_reg_weighted

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

def graphic_diagnostic(linear_reg):
    # Q-Q plot
    '''
    # Scatter plot of x vs y
    axs[0, 0].scatter(X_train, y_train)
    axs[0, 0].set_title('Scatter Plot')
    '''
    residuals = linear_reg.resid
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    # Set the background color to black
    plt.style.use('dark_background')
    """ plot color
    for ax in axs.flatten():
        ax.set_facecolor('black') 
    fig.set_facecolor('black')
    """

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












