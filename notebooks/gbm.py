import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

X = pd.read_pickle("./data/processed/X_train.pkl")
y = pd.read_pickle("./data/processed/y_train.pkl")
X_test = pd.read_pickle("./data/processed/X_test.pkl")
y_test = pd.read_pickle("./data/processed/y_test.pkl")

dtrain = xgb.DMatrix(X.astype("float"), label=y)
dtest = xgb.DMatrix(X_test.astype("float"), label=y_test)

params_constant = {
    "objective": "reg:squarederror",
    "booster": "dart",
    "lambda": 0.398779699064731,
    "alpha": 0.001604582292125471,
    "subsample": 0.4416738777730219,
    "colsample_bytree": 0.8326779169503944,
    "max_depth": 3,
    "min_child_weight": 9,
    "eta": 0.009895259749165145,
    "gamma": 2.2420356460879198e-07,
    "grow_policy": "lossguide",
    "sample_type": "uniform",
    "normalize_type": "tree",
    "rate_drop": 7.02230871007137e-05,
    "skip_drop": 1.8809036010508264e-07,
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

models = []
for kfold_round in range(5):
    kf = KFold(n_splits=5, random_state=777 + kfold_round, shuffle=True)
    for fold, (train_idx, validate_idx) in enumerate(kf.split(X)):
        X_train_sub, y_train_sub = X.iloc[train_idx], y.iloc[train_idx]
        X_validate_sub, y_validate_sub = (
            X.iloc[validate_idx],
            y.iloc[validate_idx],
        )

        dtrain = xgb.DMatrix(X_train_sub.astype("float"), label=y_train_sub)
        dvalid = xgb.DMatrix(
            X_validate_sub.astype("float"), label=y_validate_sub
        )

        watchlist = [(dtrain, "train"), (dvalid, "eval")]
        model = xgb.train(
            params_constant, dtrain, 2000, watchlist, early_stopping_rounds=50
        )

        models.append(model)

y_pred = [model.predict(dtest) for model in models]
y_pred = np.mean(y_pred, axis=0)
mean_squared_error(y_test, y_pred, squared=False)


# finalize
param = {
    "objective": "reg:squarederror",
    "booster": "dart",
    "lambda": 0.398779699064731,
    "alpha": 0.001604582292125471,
    "subsample": 0.4416738777730219,
    "colsample_bytree": 0.8326779169503944,
    "max_depth": 3,
    "min_child_weight": 9,
    "eta": 0.009895259749165145,
    "gamma": 2.2420356460879198e-07,
    "grow_policy": "lossguide",
    "sample_type": "uniform",
    "normalize_type": "tree",
    "rate_drop": 7.02230871007137e-05,
    "skip_drop": 1.8809036010508264e-07,
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
    + p9.geom_point(p9.aes("word_count_delta", "error"), alpha=0.25)
)
(
    p9.ggplot(XY_test_eval)
    + p9.geom_point(p9.aes("delete_insert_ratio", "error"), alpha=0.25)
)
