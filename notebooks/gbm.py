import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = pd.read_pickle("./data/processed/X_train.pkl")
y = pd.read_pickle("./data/processed/y_train.pkl")
X_test = pd.read_pickle("./data/processed/X_test.pkl")
y_test = pd.read_pickle("./data/processed/y_test.pkl")

dtrain = xgb.DMatrix(X.astype("float"), label=y)
dtest = xgb.DMatrix(X_test.astype("float"))

params_constant = pd.DataFrame(
    {
        "objective": "reg:squarederror",
        "booster": "dart",
        "lambda": 0.18516628343749034,
        "alpha": 0.01161424854519958,
        "subsample": 0.7513434457021767,
        "colsample_bytree": 0.6925492859363613,
        "max_depth": 3,
        "min_child_weight": 6,
        "eta": 0.009742898484514316,
        "gamma": 8.283440039691276e-07,
        "grow_policy": "lossguide",
        "sample_type": "uniform",
        "normalize_type": "forest",
        "rate_drop": 7.888155487668706e-06,
        "skip_drop": 0.04779787303400517,
    },
    index=[0],
)
params_test = pd.DataFrame(
    {"num_boost_round": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
)

params_constant = pd.concat(
    [params_constant] * params_test.shape[0]
).reset_index(drop=True)

grid_tune = pd.concat([params_constant, params_test], axis=1)

scores = []
for i in range(n_experiments := grid_tune.shape[0]):
    param = grid_tune.iloc[i, :].to_dict()
    num_boost_round = param["num_boost_round"]
    del param["num_boost_round"]
    model = xgb.train(param, dtrain, num_boost_round)

    y_pred = model.predict(dtest)

    score = {}
    score["error_summary"] = mean_squared_error(y_test, y_pred, squared=False)
    scores.append(score)

    print("iteration " + str(i) + " complete.")

scores


# finalize
param = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "lambda": 0.005,
    "alpha": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "max_depth": 3,
    "min_child_weight": 3,
    "eta": 0.01,
    "gamma": 0.015,
    "grow_policy": "lossguide",
}
num_boost_round = 1000

model = xgb.train(param, dtrain, num_boost_round)

feature_importance = pd.DataFrame.from_dict(
    model.get_score(importance_type="gain"), orient="index"
).reset_index(drop=False)
feature_importance.columns = ["feature", "score"]
feature_importance = feature_importance.sort_values("score", ascending=False)
feature_importance.head(25)

XY_test_eval = pd.concat([X_test, y_test], axis=1)
XY_test_eval["pred"] = model.predict(dtest)
XY_test_eval["error"] = XY_test_eval["y"] - XY_test_eval["pred"]
XY_test_eval["error_abs"] = XY_test_eval["error"].abs()
XY_test_eval = XY_test_eval.astype(float)

import plotnine as p9

(p9.ggplot(XY_test_eval) + p9.geom_point(p9.aes("pred", "error"), alpha=0.25))
(p9.ggplot(XY_test_eval) + p9.geom_point(p9.aes("y", "error"), alpha=0.25))
(
    p9.ggplot(XY_test_eval)
    + p9.geom_point(p9.aes("pause_time_p50", "error"), alpha=0.25)
)
(
    p9.ggplot(XY_test_eval)
    + p9.geom_point(p9.aes("delete_insert_ratio", "error"), alpha=0.25)
)
