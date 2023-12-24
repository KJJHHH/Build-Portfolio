import pandas as pd
import pickle
import statsmodels.api as sm
import numpy as np
from utils import *
from unittest import result

class Rolling_Prediction:
    def __init__(self, 
                data_week_return,  # data_preprocess
                model, 
                mode,
                transform_method):
        """
        data_week_return: data
        """
        self.data_preprocess = data_week_return
        self.model = model
        self.mode = mode
        self.start_train_date = min(data_week_return["mdate"])
        self.train, self.test = self.update_rolling_data()

        self.transform_method = transform_method  

    def update_rolling_data(self, 
                    train_size = 6,
                    first_rolling = False):
        # Update data for each rollin prediction

        if first_rolling == False:
            self.start_train_date = self.start_train_date + pd.DateOffset(weeks = 1)

        end_train_date = self.start_train_date + pd.DateOffset(weeks = self.mode['train_size'])
        start_test_date = end_train_date + pd.DateOffset(weeks = 1)
        end_test_date = start_test_date + pd.DateOffset(weeks = 1)
        train = self.data_preprocess[(self.data_preprocess["mdate"] < end_train_date) & \
                                (self.data_preprocess["mdate"] >= self.data_preprocess)]
        test = self.data_preprocess[(self.data_preprocessn["mdate"] < end_test_date) & \
                                (self.data_preprocess["mdate"] >= start_test_date)]
        return train, test

    def predict_decide_poly_kernel(train_X, train_y, test_X, test_y, mode):

            # tansformation of data    
            if mode['poly_degree'] != None:
                # transform X of train and test
                print('not okay yet')
            if mode['kernel'] == True:
                # transform X of train and test
                print('not okay yet')
            
            # check shape X of train and test
            if sm.add_constant(test_X).shape[1] != sm.add_constant(train_X).shape[1]:
                print('test_X only got 1 sample, cannot add constant with sm.add_constant. Add manually')
                test_X["const"] = 1.

            # fit model
            linear_reg = fit_linear(train_X, train_y)
            predict = linear_reg.predict(sm.add_constant(test_X))   
            return predict

    def check_data(train, test):
        # check if there is test data for each industry
        len_test = test.shape[0]

        # check if there is enough sample size for each industry
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

    def build_predict(train, test, mode):
        
        sort_industry = mode['sort_industry']

        # if sort industry
        if sort_industry == True:
            predict_result_market = pd.DataFrame()

            # seperate data by industry
            for industry in train["Industry_Eng"].unique():
                # get data for each industry
                train_1_industry = train[train["Industry_Eng"] == industry]
                test_1_industry = test[test["Industry_Eng"] == industry]    

                # check if data size is good for model
                good_data = check_data(train_1_industry, test_1_industry)
                if (good_data['len test'] != 0) and (good_data['enough sample'] == True):

                    # print(f"predicting {industry} ... with test {len(test_y)} samples")
                    train_X, train_y = split_X_y(train_1_industry)
                    test_X, test_y = split_X_y(test_1_industry)

                    # get predicted values
                    predict = predict_decide_poly_kernel(train_X, train_y, test_X, test_y, mode)
                    predict_ret = pd.DataFrame(predict, columns = ["predicted_ret"]).set_index(test_y.index)
                    predict_ret = pd.merge(test_y, predict_ret, left_index=True, right_index=True)
                    predict_result_market = pd.concat([predict_result_market, predict_ret], axis = 0)
        
        # if not sort industry
        else:
            train_X, train_y = split_X_y(train)
            test_X, test_y = split_X_y(test)

            # check if data size is good for model
            good_data = check_data(train, test)
            if (good_data['len test'] != 0) and (good_data['enough sample'] == True):
                #
                predict = predict_decide_poly_kernel(train_X, train_y, test_X, test_y, mode)
                #
                predict_ret = pd.DataFrame(predict, columns = ["predicted_ret"]).set_index(test_y.index)
                #
                predict_ret = pd.merge(test_y, predict_ret, left_index=True, right_index=True) 
                #
                predict_result_market = predict_ret
            
        return predict_result_market

    def backtest(predict_result_market, mode): 
        date = predict_result_market.reset_index()["mdate"].unique()[0]
        long_ratio = mode['long_ratio']
        selected_stock = mode['selected_stock']
        decision_trade = mode['decision_trade']
        long_ratio = mode['long_ratio']

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

    def rolling_predict(self):
        # max date
        max_date = max(self.data_preprocess["mdate"])

        # Train test period size
        train_size = self.mode['train_size'] # weeks
        start_train_date = min(self.data_preprocess["mdate"])

        end_train_date = start_train_date + pd.DateOffset(weeks = train_size)
        start_test_date = end_train_date + pd.DateOffset(weeks = 1)
        end_test_date = start_test_date + pd.DateOffset(weeks = 1)

        # Train test data
        train, test, start_train_date, start_test_date = \
            self.update_data(
                self.data_preprocess, 
                start_train_date, 
                train_size)

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
            print(f"Test: {start_test_date} with size {test.shape}")

            # check test len
            if len(test) == 0:
                # update
                train, test, start_train_date, start_test_date = \
                    update_data(data_preprocess, start_train_date, train_size)
                continue

            # train
            predict_result_market = build_predict(train, test, mode)

            # backtest
            backtest_ret = backtest(predict_result_market, mode)   
            print(f'returns for the week\n{backtest_ret}') 

            # return portfolio and market
            portfolio_ret.append(backtest_ret["port_returns"])
            market_ret.append(test["return"].mean())

            # cumulative return
            cum_portfolio *= (1 + backtest_ret["port_returns"].values[0] - 0.004)
            cum_market_hold *= (1+ test["return"].mean())
            
            # list to columns
            cum_asset = pd.DataFrame([[start_test_date, cum_portfolio, cum_market_hold]], 
                                        columns=["date", "cum_asset", "market"])
            
            # combine result of rolling rpediction
            cum_asset_portfolio_market = pd.concat([cum_asset_portfolio_market, cum_asset], axis = 0)
            
            # update
            train, test, start_train_date, start_test_date = update_data(data_preprocess, start_train_date, train_size)
            if start_test_date > max_date:
                break

        return cum_asset_portfolio_market, portfolio_ret, market_ret