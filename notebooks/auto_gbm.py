import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
import optuna

X = pd.read_pickle("./data/processed/X_train.pkl")
y = pd.read_pickle("./data/processed/y_train.pkl")
X_test = pd.read_pickle("./data/processed/X_test.pkl")
y_test = pd.read_pickle("./data/processed/y_test.pkl")

dtrain = xgb.DMatrix(X.astype("float"), label=y)
dtest = xgb.DMatrix(X_test.astype("float"))

NUM_ROUND = 1000

# helpful guidance: https://twitter.com/tunguz/status/1572642449302106112/photo/1
# for optuna experiments: fix number of rounds, adjust learning rate.
# no specification of early stopping
# then, after, adjust number of rounds/learn rate


def objective(trial):
    param = {
        "objective": "reg:squarederror",
        "booster": trial.suggest_categorical(
            "booster", ["gbtree", "gblinear", "dart"]
        ),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9, step=2)

        param["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 0.01, log=True)

        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float(
            "rate_drop", 1e-8, 1.0, log=True
        )
        param["skip_drop"] = trial.suggest_float(
            "skip_drop", 1e-8, 1.0, log=True
        )

    model = xgb.train(param, dtrain, NUM_ROUND)

    y_pred = model.predict(dtest)

    score = metrics.mean_squared_error(y_test, y_pred, squared=False)

    return score


study = optuna.create_study()
# if optuna returns nulls in y_pred, don't fail the entire study
study.optimize(
    objective, n_trials=200, catch=(ValueError,), n_jobs=-1, timeout=8 * 60 * 60
)

study.best_params
study.best_trial
study.best_value
