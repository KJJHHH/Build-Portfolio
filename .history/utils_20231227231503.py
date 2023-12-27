import pickle

def store_result(industry, model, model_config, performance_test):
    industry = model_config["industry"]
    # performance_test:
        # performance_rolling, loss_rolling, rplsw_rolling, market_rolling = performance_test
    if model == "neural network":
        model = f"{model} with number of trials {model_config['n_trials']}"

    # store result
    if model_config["train_size"] == 5:
        with open(f'result/{industry}/{model}.pk', 'wb') as f:
            pickle.dump(performance_test, f)
    else:
        with open(
            f"result/{industry}/train_size_year_{model_config['train_size']}/{model}.pk", "wb") as f:
                pickle.dump(performance_test, f)