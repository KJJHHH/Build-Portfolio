import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# other 
import gc
import pickle
from pandas.errors import DatabaseError
import warnings
warnings.filterwarnings('ignore')

def load_result(model, model_strategy): 
    if model == "neural network":
        model_ = f"{model} with number of trials {model_strategy['n_trials']}"
    else:
        model_ = model
        
    # load result
    with open(
        f"result/{industry}/result_train{model_strategy['train_size']}test{model_strategy['test_size']}/{model_}.pickle", 
        "rb") as f:
        a = pickle.load(f)
    performance_rolling, loss_rolling, rplsw_rolling, market_rolling = a
    return performance_rolling, loss_rolling, rplsw_rolling, market_rolling


industry = "automobile"
models = {1: "linear",
         2: "elastic net",
         3: "decision tree", 
         4: "random forest", 
         5: "xgboost", 
         6: "svm", 
         7: "ensemble",
         8: "neural network",
         }
print(models)
i = input("please input the model code as showed:")
model = models[int(i)]
model_strategy = {
        "model": model,     # "linear" or "decision tree" or "xgboost" or "svm" or "neural network"
        "long": 80,         # percentile
        "short": 20,        # percentile
        "train_size": 5,    # y
        "test_size": 1,     # m
        "test_year": 2021,  # start test from 2021/01
        "n_trials": 1,
        }
performance_rolling, loss_rolling, rplsw_rolling, market_rolling = load_result(model, model_strategy)
performance_mean = performance_rolling.mean().values[0]*12
performance_vol = performance_rolling.std().values[0]*(12**(1/2))
market_mean = market_rolling.mean().values[0]*12
market_vol = market_rolling.std().values[0]*(12**(1/2))

print("####################################################")
print(f"{model} performance ==================")
print(f"performance_mean: {performance_mean}, performance_vol: {performance_vol}")
print(f"market performance ==================")
print(f"market_mean: {market_mean}, market_vol: {market_vol}")
print(".......................")
print(f"{model}: sharpe ratio \
        {performance_mean/performance_vol}")
print(f"market: sharpe ratio \
        {market_mean/market_vol}")

# Trading amount
n_trading = []
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
    n_trading.append(len(long_trading) + len(short_trading))
    n_total.append(len(rplsw_rolling[rplsw_rolling["ymd"] == date]))
    trading_rate.append((len(long_trading) + len(short_trading)) / len(rplsw_rolling[rplsw_rolling["ymd"] == date]))
    date_trade.append(date)
n_long = pd.DataFrame(n_long, columns= ["n_long"]).set_index([date_trade])
n_short = pd.DataFrame(n_short, columns= ["n_short"]).set_index([date_trade])
n_trading = pd.DataFrame(n_trading, columns= ["n_trading"]).set_index([date_trade])
n_total = pd.DataFrame(n_total, columns= ["n_total"]).set_index([date_trade])
trading_rate = pd.DataFrame(trading_rate, columns= ["trading_rate"]).set_index([date_trade])

plt.figure(figsize=(15, 8))
plt.plot(n_long, label = "long")
plt.plot(n_short, label = "short")
# plt.plot(n_trading)
# plt.plot(n_total)
# plt.plot(trading_rate)
market_rolling["return"] = market_rolling["return"]*100
plt.plot(market_rolling.set_index("ymd"), label = "market")
# Shade the region where market_rolling > 0
market_rollings = market_rolling.copy()
market_rollings["return"] = np.where(market_rollings["return"] > 0, 15, -1)
plt.fill_between(market_rollings["ymd"], 0, market_rollings["return"],
                where=(market_rollings["return"] > 0), facecolor = "red", color='red', alpha=0.3, label='market > 0')
plt.legend()
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(n_long, label = "long")
plt.plot(n_short, label = "short")
# plt.plot(n_trading)
# plt.plot(n_total)
# plt.plot(trading_rate)
market_rolling["return"] = market_rolling["return"]*100
plt.plot(market_rolling.set_index("ymd"), label = "market")
# Shade the region where market_rolling > 0
market_rollings = market_rolling.copy()
market_rollings["return"] = np.where(market_rollings["return"] > 0, 15, -1)
plt.fill_between(market_rollings["ymd"], 0, market_rollings["return"],
                where=(market_rollings["return"] > 0), facecolor = "red", color='red', alpha=0.3, label='market > 0')
plt.legend()
plt.show()