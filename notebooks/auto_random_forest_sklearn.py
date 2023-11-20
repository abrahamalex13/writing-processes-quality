import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib
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

NUM_TREES = 100

def objective(trial):

    max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)

    model = RandomForestRegressor(
        n_estimators=NUM_TREES,
        max_features="sqrt",
        max_depth=max_depth,
    )
    model.fit(X, y)

    y_pred = model.predict(X_test)

    score = metrics.mean_squared_error(y_test, y_pred, squared=False)

    return score


study = optuna.create_study()
study.optimize(objective, n_trials=25)

study.best_params
study.best_trial
study.best_value

