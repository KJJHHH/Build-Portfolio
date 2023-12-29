import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# load result
# 主要的train size year: 5 years
def load_result(industry, model, model_strategy): # haven't test this function
    if model == "neural network":
        model = f"{model} with number of trials {model_strategy['n_trials']}"
        
    # load result
    if model_strategy["train_size"] == 5:
        with open(f"result/{industry}/{model}.pk", "rb") as f:
            a = pickle.load(f)
        
    else:
        with open(
            f"result/{industry}/train_size_year_{model_strategy['train_size']}/{model}.pk", "rb") as f:
                a = pickle.load(f)
                
    performance_rolling, rplsw_rolling, market_rolling = a
    return performance_rolling, rplsw_rolling, market_rolling

# result 
def models_get_result(industry, model, model_config):
    (
    performance_rolling,
    rplsw_rolling, 
    market_rolling
    ) = load_result(industry, model, model_config)
    
    return performance_rolling, market_rolling, rplsw_rolling

# Trading amount
def trading_detail(rplsw_rolling):
    n_total = []
    n_long = []
    n_short = []
    trading_rate = []
    date_trade = []
    rplsw_rolling = rplsw_rolling.reset_index()
    for date in rplsw_rolling["ymd"].unique():
        short_trading = rplsw_rolling[(rplsw_rolling["pred_ls"] == -1) & (rplsw_rolling["ymd"] == date)]
        long_trading = rplsw_rolling[(rplsw_rolling["pred_ls"] == 1) & (rplsw_rolling["ymd"] == date)]
        n_long.append(len(long_trading))
        n_short.append(len(short_trading))
        n_total.append(len(rplsw_rolling[rplsw_rolling["ymd"] == date]))
        trading_rate.append((len(long_trading) + len(short_trading)) / len(rplsw_rolling[rplsw_rolling["ymd"] == date]))
        date_trade.append(date)
    n_long = pd.DataFrame(n_long, columns= ["n_long"]).set_index([date_trade])
    n_short = pd.DataFrame(n_short, columns= ["n_short"]).set_index([date_trade])
    n_total = pd.DataFrame(n_total, columns= ["n_total"]).set_index([date_trade])
    trading_rate = pd.DataFrame(trading_rate, columns= ["trading_rate"]).set_index([date_trade])
    return n_long, n_short, trading_rate

# pipeline and plot
def plot_model_result(industry, model, model_config):
    """
    1. Add transaction fee to the final returns 
    2. Get trading detail inculde number of long and short, and trading rate (n to trade/all stock)
    3. plot
    """
    performance_rolling, rplsw_rolling, market_rolling = \
        models_get_result(industry, model, model_config)

    # transcation fee
    performance_rolling -= performance_rolling
    
    # trade detail
    n_long, n_short, trading_rate = trading_detail(rplsw_rolling)
    result = performance_rolling.\
                            merge(market_rolling, on="ymd", how='outer').\
                            set_index(["ymd"])
    
    # cumulative asset for port and market
    market_cum_asset = np.cumprod(1 + market_rolling["return"])
    portfolio_cum_asset = np.cumprod(1 - 0.004 + performance_rolling["performance"])
    print(market_cum_asset.set_index(market_rolling.index))
    print(portfolio_cum_asset)

    # mean, col for port and market
    performance_mean = performance_rolling.mean().values[0]*12
    performance_vol = performance_rolling.std().values[0]*(12**(1/2))
    market_mean = market_rolling.mean().values[0]*12
    market_vol = market_rolling.std().values[0]*(12**(1/2))

    """
    # prints
    # print(f"{model} performance ==================")
    # print(f"performance_mean: {performance_mean}, performance_vol: {performance_vol}")
    # print(".......................")
    # print(f"{model}: sharpe ratio \
    #         {performance_mean/performance_vol}")    
    """
    print(f"market performance ==================")
    print(f"market_mean: {market_mean}, market_vol: {market_vol}")
    print(f"market: sharpe ratio {market_mean/market_vol}")

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    axs[0].plot(portfolio_cum_asset, label=f"portfolio cum ret")
    axs[0].plot(market_cum_asset, label = 'market cum ret')
    axs[0].set_title(f"{model} cumulative asset") 
    axs[0].text(18900, 1.2, 
        f'''
        porfolio mean return {performance_mean: .5f}
        porfolio vol {performance_vol: .5f}
        porfolio sharpe {performance_mean/performance_vol: .5f}
        ''', 
        fontsize=12, 
        color='red')
    # plt.text(3, 7, f'porfolio vol {performance_vol}', fontsize=12, color='red')
    # plt.text(3, 7, f'porfolio sharpe {performance_mean/performance_vol}', fontsize=12, color='red')

    # Shade the region where market_rolling > 0
    market_rolling["return"] = market_rolling["return"]*100    
    market_rollings = market_rolling.copy()
    market_rollings["return"] = np.where(market_rollings["return"] > 0, 15, -1)
    axs[1].plot(n_long, label = "number of long")
    axs[1].plot(n_short, label = "number of short")
    axs[1].plot(trading_rate, label = "trade rate")
    axs[1].set_title(f"{model} trading amount")
    axs[1].fill_between(market_rollings["ymd"], 0, market_rollings["return"],
                    where=(market_rollings["return"] > 0), 
                    facecolor = "red", 
                    color='red', alpha=0.3, label='market > 0')
    # plt.legend()
    plt.tight_layout()
    plt.show()
    