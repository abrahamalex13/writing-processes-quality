import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
from collections import OrderedDict
from sklearn.model_selection import train_test_split

X = pd.read_pickle("./data/processed/X_train.pkl")
y = pd.read_csv("./data/external/train_scores.csv")
y.index = y['id']
y = y.rename(columns={'score': 'y'})
y = y.drop(columns='id')
XY = pd.merge(X, y, how='left', left_index=True, right_index=True)
y = XY['y']
X = XY.drop(columns='y')

X, X_test, y, y_test = train_test_split(
    X, y, test_size=0.33, random_state=777
    )

dtrain = xgb.DMatrix(X.astype("float"), label=y)
dtest = xgb.DMatrix(X_test.astype("float"))

grid_tune = pd.DataFrame(
    {
        "objective": "reg:squarederror",
        "num_boost_round": [100],
        "eta": [0.01],
        "max_depth": [1],
        "subsample": [0.5],
        "lambda": [0],
        "alpha": [0]
    }
)

grid_tune = grid_tune.iloc[np.repeat(0, 5), :].reset_index(drop=True)
grid_tune["num_boost_round"] = [
    100,
    500,
    750,
    1000,
    2000
]
# grid_tune["max_depth"] = [1, 2, 3, 5]

scores = []
for i in range(grid_tune.shape[0]):

    param_all = grid_tune.iloc[i, :].to_dict()
    param = param_all.copy()
    del param["num_boost_round"]
    num_boost_round = param_all["num_boost_round"]
    model = xgb.train(param, dtrain, num_boost_round)

    y_pred = model.predict(dtest)

    score = {}
    score["error_summary"] = metrics.mean_squared_error(
        y_test, y_pred, squared=False
    )
    scores.append(score)

    print("iteration " + str(i) + " complete.")

scores


# finalize
param = {
    "objective": "reg:squarederror",
    "eta": 0.01,
    "max_depth": 1,
    "subsample": 0.5,
    "lambda": 0,
    "alpha": 0,
}
num_boost_round = 1000

model = xgb.train(param, dtrain, num_boost_round)

feature_importance = pd.DataFrame.from_dict(
    model.get_score(importance_type="gain"), orient="index"
).reset_index(drop=False)
feature_importance.columns = ["feature", "score"]
feature_importance = feature_importance.sort_values("score", ascending=False)