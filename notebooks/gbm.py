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
dtest = xgb.DMatrix(X_test.astype("float"), label=y_test)

params_constant = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "lambda": 0.40126230344517827,
    "alpha": 0.7371523926815153,
    "subsample": 0.7833463483449391,
    "colsample_bytree": 0.3485745490154163,
    "max_depth": 5,
    "min_child_weight": 9,
    "eta": 0.005245686094970051,
    "gamma": 0.6533572521917524,
    "grow_policy": "lossguide",
}
# params_constant = pd.DataFrame(
#     {
#         "objective": "reg:squarederror",
#         "booster": "gbtree",
#         "lambda": 0,
#         "alpha": 0.66,
#         "subsample": 0.85,
#         "colsample_bytree": 0.9,
#         "max_depth": 7,
#         "min_child_weight": 10,
#         "eta": 0.006,
#         "gamma": 0,
#         "grow_policy": "lossguide",
#     },
#     index=[0],
# )
# params_test = pd.DataFrame({"num_boost_round": [800, 900, 1000, 1100, 1200]})
# params_constant = pd.concat(
#     [params_constant] * params_test.shape[0]
# ).reset_index(drop=True)

# grid_tune = pd.concat([params_constant, params_test], axis=1)

# xgb.train early stopping example:
# https://www.kaggle.com/code/tunguz/xgboost-one-step-ahead-lb-0-519/script

# scores = []
# for i in range(n_experiments := grid_tune.shape[0]):
#     param = grid_tune.iloc[i, :].to_dict()
#     num_boost_round = param["num_boost_round"]
#     del param["num_boost_round"]
#     model = xgb.train(param, dtrain, num_boost_round)

#     y_pred = model.predict(dtest)

#     score = {}
#     score["error_summary"] = mean_squared_error(y_test, y_pred, squared=False)
#     scores.append(score)

#     print("iteration " + str(i) + " complete.")

# scores

watchlist = [(dtrain, "train"), (dtest, "eval")]
model = xgb.train(
    params_constant, dtrain, 2000, watchlist, early_stopping_rounds=50
)


# finalize
param = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "lambda": 0.40126230344517827,
    "alpha": 0.7371523926815153,
    "subsample": 0.7833463483449391,
    "colsample_bytree": 0.3485745490154163,
    "max_depth": 5,
    "min_child_weight": 9,
    "eta": 0.005245686094970051,
    "gamma": 0.6533572521917524,
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
