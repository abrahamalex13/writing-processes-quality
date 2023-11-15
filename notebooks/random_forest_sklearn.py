import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib

X = pd.read_pickle("./data/processed/X_train.pkl")
y = pd.read_csv("./data/external/train_scores.csv")
y.index = y["id"]
y = y.rename(columns={"score": "y"})
y = y.drop(columns="id")
XY = pd.merge(X, y, how="left", left_index=True, right_index=True)
y = XY["y"]
X = XY.drop(columns="y")

X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=777)


step_n_estimators = 100
n_iter = 10
tune_grid = pd.DataFrame(
    {"n_trees": [step_n_estimators * (i + 1) for i in range(n_iter)]}
)
scores = []

model = RandomForestRegressor(
    n_estimators=step_n_estimators,
    max_features="sqrt",
    max_depth=None,
)
for i in range(tune_grid.shape[0]):
    model.set_params(n_estimators=tune_grid["n_trees"].iloc[i])

    model.fit(X, y.values)

    y_pred = model.predict(X_test)

    # trial-specific score dict may store multiple metrics
    score = {}

    score["error_summary"] = metrics.mean_squared_error(
        y_test, y_pred, squared=False
    )

    scores.append(score)

    print("Experiment " + str(i) + " complete.")

scores = pd.DataFrame(scores)
results = pd.concat([tune_grid, scores], axis=1)
# results.plot(x="n_trees", y="error_summary", kind="line")
