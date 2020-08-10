import pickle

import pandas as pd

import tensorflow as tf

with open("./models/wide_and_deep/w_d_keras.json", "r") as json_file:
    model = tf.keras.models.model_from_json(json_file.read())
model.load_weights("./models/wide_and_deep/w_d_keras")

with open("./models/one_hot_encoder.b", 'rb') as f:
    enc = pickle.load(f)


# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


test_file = "./data/adult.test"

print('Preparing data...')
# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

org_linear_inputs = df_train[CATEGORICAL_COLUMNS]

linear_inputs = enc.transform(org_linear_inputs).todense()

dnn_inputs = df_train[CONTINUOUS_COLUMNS]

y = df_train["income_bracket"].apply(lambda x: '>50K' in x).astype(int)

y_pred = model.predict([linear_inputs, dnn_inputs])
import numpy as np
print(np.min(y_pred), np.max(y_pred), np.mean(y_pred))

from sklearn.metrics import roc_auc_score

auc_score = roc_auc_score(y, y_pred)
print("Pure w&d auc: ", auc_score)
