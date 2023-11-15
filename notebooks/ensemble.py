import pandas as pd
import numpy as np
import xgboost as xgb
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skranger.ensemble import RangerForestRegressor
import optuna

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
param = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "lambda": 0.9,
    "alpha": 0.01,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "max_depth": 3,
    "min_child_weight": 5,
    "eta": 0.01,
    "grow_policy": "depthwise",
    "gamma": 0.00025,
}
num_boost_round = 750
model = xgb.train(param, dtrain, num_boost_round)
pred_gbm = model.predict(dtest)


model = RangerForestRegressor(
    n_estimators=500,
    mtry=math.floor(math.sqrt(X.shape[1])),
    max_depth=0,
)
model.fit(X, y.values)
pred_rf = model.predict(X_test)


def objective(trial):
    param = [
        trial.suggest_uniform("w1", 0, 1),
        trial.suggest_uniform("w2", 0, 1),
    ]

    w = [x / sum(param) for x in param]

    y_pred = w[0] * pred_gbm + w[1] * pred_rf

    score = mean_squared_error(y_test, y_pred, squared=False)

    return score


study = optuna.create_study()
study.optimize(objective, n_trials=200)

study.best_params
best_params_normalized = list(study.best_params.values())
best_params_normalized = [
    x / sum(best_params_normalized) for x in best_params_normalized
]
study.best_trial
study.best_value
