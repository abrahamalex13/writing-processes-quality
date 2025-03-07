import pandas as pd
import numpy as np
import xgboost as xgb
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import optuna
import tensorflow as tf
from tensorflow import keras

from src.data.extract_scrub_essay_text import (
    extract,
    scrub_activity,
    scrub_text_change,
    concatenate_essay_from_logs,
)


X = pd.read_pickle("./data/processed/X_train.pkl")
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
y = pd.read_pickle("./data/processed/y_train.pkl")

X_test = pd.read_pickle("./data/processed/X_test.pkl")
X_test_std = std_scaler.transform(X_test)
y_test = pd.read_pickle("./data/processed/y_test.pkl")

dtrain = xgb.DMatrix(X.astype("float"), label=y)
dtest = xgb.DMatrix(X_test.astype("float"))
param = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "lambda": 2.2864503265614494e-07,
    "alpha": 3.291236167053757e-05,
    "subsample": 0.5145373136184048,
    "colsample_bytree": 0.44401075818853125,
    "max_depth": 3,
    "min_child_weight": 6,
    "eta": 0.00917371721047756,
    "gamma": 8.217276838858242e-05,
    "grow_policy": "depthwise",
}
num_boost_round = 1000
model = xgb.train(param, dtrain, num_boost_round)
pred_gbm = model.predict(dtest)


model = RandomForestRegressor(
    n_estimators=1000,
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
essays = pd.concat(
    [concatenate_essay_from_logs(x) for x in X_train_logs], axis=0
)
# keras TextVectorization does not recognize emdash as punctuation
essays["essay"] = essays["essay"].str.replace("—", " ")

essays = essays.set_index("id")
essays_train = (
    pd.merge(y, essays, "inner", left_index=True, right_index=True)
    .drop(columns="y")
    .to_numpy()
)
essays_test = (
    pd.merge(y_test, essays, "inner", left_index=True, right_index=True)
    .drop(columns="y")
    .to_numpy()
)

BATCH_SIZE = 32

# in tf Dataset structure, one element is one X-y pair
essays_y_train_nn = tf.data.Dataset.from_tensor_slices(
    (essays_train, y.to_numpy())
)
essays_train_nn = essays_y_train_nn.map(lambda x, y: x)

essays_y_test_nn = tf.data.Dataset.from_tensor_slices(
    (essays_test, y_test.to_numpy())
)
essays_test_nn = essays_y_test_nn.map(lambda x, y: x)

text_vectorization = tf.keras.layers.TextVectorization(
    # with anonymized text, downscale recommended vocabulary size by magnitude
    max_tokens=20000,
    # standardize="lower_and_strip_punctuation",
    # other models' results imply, punctuation usage is important
    standardize="lower",
    split="whitespace",
    ngrams=4,
    output_mode="count",
)

text_vectorization.adapt(essays_train_nn)
# values = text_vectorization.get_vocabulary()

# XY_train_nn = essays_y_train_nn.map(
#     lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
# )
# X_train_nn = XY_train_nn.map(lambda x, y: x)

# XY_test_nn = essays_y_test_nn.map(
#     lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
# )
# X_test_nn = XY_test_nn.map(lambda x, y: x)

# X_writing_proc_train_nn = tf.data.Dataset.from_tensor_slices(X)
# y_writing_proc_train_nn = tf.data.Dataset.from_tensor_slices(y)

n_tokens = text_vectorization.vocabulary_size()
n_features = X.shape[1]


# HYPOTHESIS: this keras functional api implementation requires all numpy inputs
# Batch-type, map-type tf Dataset input types have failed

input_essay = keras.Input(shape=(1,), dtype=tf.string, name="essay_text")
vectorized_essay = text_vectorization(input_essay)
supervised_pca_essay = keras.layers.Dense(64, activation="relu")(
    vectorized_essay
)

input_writing_process = keras.Input(shape=(n_features,), name="writing_process")

features = keras.layers.Concatenate()(
    [supervised_pca_essay, input_writing_process]
)
# 256 too many, 128 sees MSE go to ~0.49
features = keras.layers.Dense(256, activation="relu")(features)
features = keras.layers.Dropout(0.5)(features)

outputs = keras.layers.Dense(1)(features)

model = keras.Model(
    inputs=[input_essay, input_writing_process], outputs=outputs
)

model.compile(optimizer="rmsprop", loss="mean_squared_error")

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=50
)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="tmp/checkpoint_nn",
    monitor="val_loss",
    mode="min",
    save_weights_only=True,
    save_best_only=True,
)
model.fit(
    [essays_train, X_std],
    y.to_numpy(),
    validation_data=([essays_test, X_test_std], y_test.to_numpy()),
    batch_size=BATCH_SIZE,
    epochs=200,
    callbacks=[early_stopping_callback, model_checkpoint_callback],
)

model.load_weights("checkpoint_nn")
pred_nn = model.predict([XY_test_nn, X_test.to_numpy()])[:, 0]


def objective(trial):
    param = [
        trial.suggest_uniform("w1", 0, 1),
        trial.suggest_uniform("w2", 0, 1),
        # trial.suggest_uniform("w3", 0, 1),
    ]

    w = [x / sum(param) for x in param]

    y_pred = (
        w[0] * pred_gbm
        + w[1] * pred_rf
        # + w[2] * pred_nn
    )

    score = mean_squared_error(y_test, y_pred, squared=False)

    return score


study = optuna.create_study()
study.optimize(objective, n_trials=50)

study.best_params
best_params_normalized = list(study.best_params.values())
best_params_normalized = [
    x / sum(best_params_normalized) for x in best_params_normalized
]
best_params_normalized
study.best_trial
study.best_value
