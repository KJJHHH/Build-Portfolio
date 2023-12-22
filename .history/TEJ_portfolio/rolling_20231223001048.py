def rolling_predict(data_preprocess, mode):
    # max date
    import pickle
    max_date = max(data_preprocess["mdate"])

    # Train test period size
    train_size = 52 # weeks
    start_train_date = min(data_preprocess["mdate"])
    end_train_date = start_train_date + pd.DateOffset(weeks = train_size)
    start_test_date = end_train_date + pd.DateOffset(weeks = 1)
    end_test_date = start_test_date + pd.DateOffset(weeks = 1)

    # Train test data
    train, test, start_train_date, start_test_date = \
        update_data(
            data_preprocess, 
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