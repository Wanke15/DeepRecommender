import pickle
import numpy as np

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

train_file = "../wide_and_deep/data/adult.train"
test_file = "../wide_and_deep/data/adult.test"

print('Preparing data...')
# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)

final_train_inputs = df_train[CATEGORICAL_COLUMNS]

y = df_train["income_bracket"].apply(lambda x: '>50K' in x).astype(int)
print(df_train["income_bracket"].value_counts())
exit()

# data = load_breast_cancer();data.data;data.target
X_train, X_eval, y_train, y_eval = train_test_split(final_train_inputs, y, test_size=0.2,
                                                    random_state=27, stratify=y)
#############
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True)

final_test_inputs = df_test[CATEGORICAL_COLUMNS]

y_test = df_test["income_bracket"].apply(lambda x: '>50K' in x).astype(int)

from catboost import CatBoostClassifier

categorical_features_indices = [_ for _ in range(final_train_inputs.shape[1])]

model = CatBoostClassifier(iterations=500,
                           early_stopping_rounds=50,
                           cat_features=categorical_features_indices,
                           learning_rate=0.01,
                           loss_function='Logloss',
                           logging_level='Verbose',
                           custom_metric=['AUC'],
                           # class_weights=[0.25, 0.75]
                           )

model.fit(final_train_inputs, y, eval_set=(final_test_inputs, y_test), plot=True)
