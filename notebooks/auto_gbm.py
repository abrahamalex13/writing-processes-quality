import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
import optuna
from sklearn.model_selection import train_test_split

X = pd.read_pickle("./data/processed/X_train.pkl")
y = pd.read_csv("./data/external/train_scores.csv")
y.index = y["id"]
y = y.rename(columns={"score": "y"})
y = y.drop(columns="id")
XY = pd.merge(X, y, how="left", left_index=True, right_index=True)
y = XY["y"]
X = XY.drop(columns="y")

X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=777)

dtrain = xgb.DMatrix(X.astype("float"), label=y)
dtest = xgb.DMatrix(X_test.astype("float"))


def objective(trial):
    param = {
        "objective": "reg:squarederror",
        "booster": trial.suggest_categorical(
            "booster", ["gbtree", "gblinear", "dart"]
        ),
        "num_boost_round": 10,
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)

        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)

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

    model = xgb.train(param, dtrain)

    y_pred = model.predict(dtest)

    score = metrics.mean_squared_error(y_test, y_pred, squared=False)

    return score


study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params