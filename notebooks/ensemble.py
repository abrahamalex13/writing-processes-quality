import pandas as pd
import numpy as np
import xgboost as xgb
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import optuna
import tensorflow as tf
from tensorflow import keras

from transform_features_nn import (
    extract,
    scrub_activity,
    scrub_text_change,
    concatenate_essay_from_logs,
)


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
    "lambda": 0.0005,
    "alpha": 0.00025,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "max_depth": 3,
    "min_child_weight": 3,
    "eta": 0.01,
    "gamma": 0.1,
    "grow_policy": "lossguide",
}
num_boost_round = 750
model = xgb.train(param, dtrain, num_boost_round)
pred_gbm = model.predict(dtest)


model = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    max_depth=None,
)
model.fit(X, y.values)
pred_rf = model.predict(X_test)


PATH_TRAIN_LOGS = "./data/external/train_logs.csv"
X_train_logs = extract(PATH_TRAIN_LOGS)
X_train_logs = scrub_activity(X_train_logs)
X_train_logs = scrub_text_change(X_train_logs)

X_train_logs = [x for _, x in X_train_logs.groupby("id")]
essays_text = pd.concat(
    [concatenate_essay_from_logs(x) for x in X_train_logs], axis=0
)
# keras TextVectorization does not recognize emdash as punctuation
essays_text["essay"] = essays_text["essay"].str.replace("—", " ")

essays_text = essays_text.set_index("id")
X = (
    pd.merge(y, essays_text, "inner", left_index=True, right_index=True)
    .drop(columns="y")
    .to_numpy()
)
X_test = (
    pd.merge(y_test, essays_text, "inner", left_index=True, right_index=True)
    .drop(columns="y")
    .to_numpy()
)

BATCH_SIZE = 32

# in tf Dataset structure, one element is one X-y pair
XY_train = tf.data.Dataset.from_tensor_slices((X, y.to_numpy())).batch(
    BATCH_SIZE
)
X_train = XY_train.map(lambda x, y: x)

XY_test = tf.data.Dataset.from_tensor_slices((X_test, y_test.to_numpy())).batch(
    BATCH_SIZE
)
X_test = XY_test.map(lambda x, y: x)

text_vectorization = tf.keras.layers.TextVectorization(
    # with anonymized text, downscale recommended vocabulary size by magnitude
    max_tokens=20000,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ngrams=2,
    output_mode="tf_idf",
)

text_vectorization.adapt(X_train)
# values = text_vectorization.get_vocabulary()

tfidf_XY_train = XY_train.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

tfidf_XY_test = XY_test.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

n_tokens = text_vectorization.vocabulary_size()

inputs = keras.Input(shape=(n_tokens,))
x = keras.layers.Dense(64, activation="relu")(inputs)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="mean_squared_error")

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10
)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint_nn",
    monitor="val_loss",
    mode="min",
    save_weights_only=True,
    save_best_only=True,
)
model.fit(
    tfidf_XY_train.cache(),
    validation_data=tfidf_XY_test.cache(),
    epochs=100,
    callbacks=[early_stopping_callback, model_checkpoint_callback],
)

model.load_weights("checkpoint_nn")
pred_nn = model.predict(tfidf_XY_test)[:, 0]


def objective(trial):
    param = [
        trial.suggest_uniform("w1", 0, 1),
        trial.suggest_uniform("w2", 0, 1),
        trial.suggest_uniform("w3", 0, 1),
    ]

    w = [x / sum(param) for x in param]

    y_pred = w[0] * pred_gbm + w[1] * pred_rf + w[2] * pred_nn

    score = mean_squared_error(y_test, y_pred, squared=False)

    return score


study = optuna.create_study()
study.optimize(objective, n_trials=200)

study.best_params
best_params_normalized = list(study.best_params.values())
best_params_normalized = [
    x / sum(best_params_normalized) for x in best_params_normalized
]
best_params_normalized
study.best_trial
study.best_value
