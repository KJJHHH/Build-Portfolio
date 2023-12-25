import pandas as pd
import pickle
import statsmodels.api as sm
import numpy as np
from utils import *
from unittest import result
import os

class Rolling_Prediction:
    def __init__(self, 
                data_week_return,  # data_clean
                train_size,
                model, 
                mode,
                transform_method):
        super(Rolling_Prediction).__init__()
        """
        data_week_return: data_clean, withput transforming data
        model: str, model to fit. 'ols', 'wls'
        mode: dictionary, setting for rolling prediction
        transform_method: dictionary, method to transform data 
        train_size: ~
        """
        self.data_clean = data_week_return
        self.model = model
        self.mode = mode
        self.train_size = train_size
        self.result_path = f'result/{self.model}'

        # Check if the directory exists. If not, create
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print(f"Folder created successfully at: {self.result_path}")
        else:
            print(f"Folder already exists at: {self.result_path}")


        self.start_train_date = min(data_week_return["mdate"])
        self.start_test_date = 0
        self.train, self.test = 0, 0
        self.update_rolling_data(first_rolling=True)
        self.max_date = max(self.data_clean["mdate"])

        self.transform_method = transform_method  
        
        """
        ## feature engineering
        self.feature_eng = feature_engineer()
        X_scale_winsor = self.feature_eng.transform_standard_winsor(X_train, scale_method='Standard', winsor=True)
        X_poly_selected_corr = self.feature_eng.transform_poly_selected_pearsonr(X_scale_winsor, y_train)
        X_pca = self.feature_eng.transform_pca(X_poly_selected_corr, var = 100)
        X_selected_bic = self.feature_eng.selected_bic(X_pca, y_train)

        ## build linear models
        linear = build_ols(sm.add_constant(X_selected_bic), y_train, "HC0")
        linear = build_wls(sm.add_constant(X_selected_bic), y_train, "HC0")
        """
        self.feature_eng = feature_engineer()

    def update_rolling_data(self, first_rolling = False):
        # Update data for each rollin prediction

        if first_rolling == False:
            self.start_train_date += pd.DateOffset(weeks = 1)

        end_train_date = self.start_train_date + pd.DateOffset(weeks = self.train_size)
        start_test_date = end_train_date + pd.DateOffset(weeks = 1)
        end_test_date = start_test_date + pd.DateOffset(weeks = 1)
        self.train = self.data_clean[(self.data_clean["mdate"] < end_train_date) & \
                                (self.data_clean["mdate"] >= self.start_train_date)]
        self.test = self.data_clean[(self.data_clean["mdate"] < end_test_date) & \
                                (self.data_clean["mdate"] >= start_test_date)]
        # start test date
        self.start_test_date = start_test_date

    def feature_engineering(self, X_train, X_test, y_train):
        # Check self.feature_eng for how to use
        """
        X_train,
        X_test, 
        """

        # scale and winsro
        X_scale_winsor = self.feature_eng.transform_standard_winsor(X_train, scale_method='Standard', winsor=True)
        X_test = self.feature_eng.transform_standard_winsor(X_test, scale_method='Standard', winsor=True)

        X_poly_selected_corr, selected_col_corr = self.feature_eng.transform_poly_selected_pearsonr(X_scale_winsor, y_train)
        X_test = self.feature_eng.transform_poly_selected_pearsonr(X_test, None, selected_col_corr)

        X_pca, X_test = self.feature_eng.transform_pca(X_poly_selected_corr, X_test, var = 100)

        X_selected_bic = self.feature_eng.selected_bic(X_pca, y_train)
        X_test = self.feature_eng.selected_bic(X_test, None, selected_col_corr)

        X_train = X_selected_bic
        
        return X_train, X_test


    def transform_poly_kernel_addconstant(self, train_X, train_y, test_X, test_y):
        ###########################################
        # 1. transform: scale, winsor
        # 2. transform: kernel
        # 3. chech if add_constant work in test_X
            # if only 1 sample than don't work; add manually
        ###########################################
        
        # transformation of data
        train_X, test_X = self.trans.transform(train_X, test_X)

        # tansformation of data    
        if self.mode['poly_degree'] != None:
            # transform X of train and test
            print('not okay yet')
        if self.mode['kernel'] == True:
            # transform X of train and test
            print('not okay yet')
        
        # add constant
        train_X = sm.add_constant(train_X)
        test_X = sm.add_constant(test_X)
        if test_X.shape[1] != train_X.shape[1]:
            print('test_X only got 1 sample, cannot add constant with sm.add_constant. Add manually')
            test_X["const"] = 1.

        return train_X,train_y,test_X,test_y
    
    def fit_model(self, train_X, train_y, test_X, test_y):
        # fit model
        if self.model == 'ols':
            fit_reg = build_ols(train_X, train_y)
        elif self.model == 'wls':
            fit_reg = build_wls(train_X, train_y)
        
        # predict
        predict = fit_reg.predict(sm.add_constant(test_X)) 
        return predict  

    def check_data(self, train, test):
        ###########################################
        # 1. check if there is test data for each industry
        # 2. check if there is enough sample size for each industry
        ###########################################
        len_test = test.shape[0]
        enough_sample = train.shape[0] > train.shape[1]

        # check
        if len_test == 0:
            print('No test data for the date ?. Skip this industry ad the date')
        if enough_sample == False:
            print(f'too small smaple size for industry ?. Skip this industry ad the date')

        # return
        return {
            'len test': len_test, 
            'enough sample': enough_sample
            }

    def predicting(self):
        ###########################################
        # 1. dicide if sort industry when predicting
        ###########################################

        sort_industry = self.mode['sort_industry']

        # if sort industry
        if sort_industry == True:
            predict_result_market = pd.DataFrame()

            # seperate data by industry
            for industry in self.train["Industry_Eng"].unique():
                # get data for each industry
                train_1_industry = self.train[self.train["Industry_Eng"] == industry]
                test_1_industry = self.test[self.test["Industry_Eng"] == industry]    

                # check if data size (of different industry)
                # 1. length of test
                # 2. enough sample size
                good_data = self.check_data(train_1_industry, test_1_industry)
                if (good_data['len test'] == 0) or (good_data['enough sample'] == False):
                    continue
                
                # split data to X, y
                train_X, train_y = split_X_y(train_1_industry)
                test_X, test_y = split_X_y(test_1_industry)
                

                # get predicted values
                train_X, train_y, test_X, test_y = \
                    self.transform_poly_kernel_addconstant(train_X, train_y, test_X, test_y)
                
                # fit model
                predict = self.fit_model(train_X, train_y, test_X, test_y)
                
                # make the prediction to dataframe
                predict_ret = pd.DataFrame(predict, columns = ["predicted_ret"])\
                    .set_index(test_y.index)
                
                # merge real and prediction values
                predict_ret = pd.merge(test_y, predict_ret, left_index=True, right_index=True)

                # concat differ industry result
                predict_result_market = pd.concat([predict_result_market, predict_ret], axis = 0)
        
        # if not sort industry
        else:
            train_X, train_y = split_X_y(self.train)
            test_X, test_y = split_X_y(self.test)

            # check if data size (of different industry) is good for model
            good_data = self.check_data(self.train, self.test)
            if (good_data['len test'] != 0) and (good_data['enough sample'] == True):
                # transformation of data
                train_X, test_X = self.trans.transform(train_X, test_X)

                # get prediction
                train_X, train_y, test_X, test_y = \
                    self.transform_poly_kernel_addconstant(train_X, train_y, test_X, test_y)

                # fit model
                predict = self.fit_model(train_X, train_y, test_X, test_y)
                
                # make the prediction to dataframe
                predict_ret = pd.DataFrame(predict, columns = ["predicted_ret"])\
                    .set_index(test_y.index)
                
                # merge real and prediction values
                predict_ret = pd.merge(test_y, predict_ret, left_index=True, right_index=True) 
                predict_result_market = predict_ret

            else:
                print('----------------------')
                print('warning: bad data')
                print(f"predicting {industry} ... with train {test_X.shape[0]} samples")
                print(f"predicting {industry} ... with test {len(test_y)} samples")
                print('----------------------')
            
        return predict_result_market

    def rolling_predict(self):
        ################################################################
        # Initialise of asset
        cum_portfolio = 1
        cum_market_hold = 1
        portfolio_ret = []
        market_ret = []
        cum_asset_portfolio_market = pd.DataFrame()

        # Rolling Prediction
        while True:
            print("="*80)
            print(f"Test: {self.start_test_date} with size {self.test.shape}")

            # check test len
            if len(self.test) == 0:
                self.update_rolling_data(first_rolling=False)
                continue
            

            # train
            predict_result_market = self.predicting()

            # backtest
            backtest_ret = self.portfolio_period_return(predict_result_market)   
            print(f'returns for the week\n{backtest_ret}') 

            # return portfolio and market
            portfolio_ret.append(backtest_ret["port_returns"][0] - 0.004)
            market_ret.append(self.test["return"].mean())

            # cumulative return
            cum_portfolio *= (1 + backtest_ret["port_returns"].values[0] - 0.004)
            cum_market_hold *= (1+ self.test["return"].mean())
            
            # list to columns
            cum_asset = pd.DataFrame([[self.start_test_date, cum_portfolio, cum_market_hold]], 
                                        columns=["date", "cum_asset", "market"])
            
            # combine result of rolling rpediction
            cum_asset_portfolio_market = pd.concat([cum_asset_portfolio_market, cum_asset], axis = 0)
            
            # update
            self.update_rolling_data(first_rolling=False)
            if self.start_test_date > self.max_date:
                break
                
        # result to store
        result = (cum_asset_portfolio_market, portfolio_ret, market_ret)
        self.store_result(result)     
        
        return cum_asset_portfolio_market, portfolio_ret, market_ret

    def portfolio_period_return(self, predict_result_market): 
        
        date = predict_result_market.reset_index()["mdate"].unique()[0]
        long_ratio = self.mode['long_ratio']
        selected_stock = self.mode['selected_stock']
        decision_trade = self.mode['decision_trade']
        long_ratio = self.mode['long_ratio']

        # Decide when to trade, Got stock to be traded
        if selected_stock == 'percent':
            long = np.percentile(predict_result_market["predicted_ret"], decision_trade)
            short = np.percentile(predict_result_market["predicted_ret"], (100 - decision_trade))
            long_stock = predict_result_market[predict_result_market["predicted_ret"] >= long]
            short_stock = predict_result_market[predict_result_market["predicted_ret"] <= short]

        if selected_stock == 'top':
            long_stock = predict_result_market.sort_values('predicted_ret', ascending=False)\
                .head(decision_trade)
            short_stock = predict_result_market.sort_values('predicted_ret', ascending=True)\
                .head(decision_trade)

        # Decide weights
        long_stock["predicted_ret"] = long_stock["predicted_ret"] + abs(min(long_stock["predicted_ret"]))
        long_stock["weight"] = long_stock["predicted_ret"]/sum(long_stock["predicted_ret"]) * long_ratio
        short_stock["predicted_ret"] = short_stock["predicted_ret"] - abs(max(short_stock["predicted_ret"]))
        short_stock["weight"] = short_stock["predicted_ret"]/sum(short_stock["predicted_ret"]) * (1-long_ratio)
        total_weights = sum(long_stock["weight"]) + sum(short_stock["weight"])

        # Portfolio return
        returns = sum(long_stock["return"] * long_stock["weight"]) - sum(short_stock["return"] * short_stock["weight"])
        backtest_ret = pd.DataFrame([[date, returns]], columns=["date", "port_returns"])

        # Print results
        print(f"Check Sum Weights: {total_weights}")
        '''
        print(f"Backtest Returns: {returns}")
        print(f"Long Stock: {len(long_stock.reset_index()['coid'])}, \
            {long_stock.reset_index()['coid'].values}")
        print(f"Short Stock: {len(short_stock.reset_index()['coid'])}, \
            {short_stock.reset_index()['coid'].values}")
        '''
        
        return backtest_ret

    
    def store_result(self, result):
        """
        result: ()
        """
        # get filename and save result
        filename = ''
        for i in list(self.mode.keys()):
            value = self.mode[i]
            filename += i + '-' + str(value) + ' '
        with open(f'result/{self.model}/mode_[{filename}].pk', 'wb') as f:
            pickle.dump((self.mode, result), f)
        print('Seccessfully stored')
