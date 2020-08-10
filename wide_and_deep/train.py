import tensorflow as tf
import pandas as pd

# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

train_file = "./data/adult.train"
test_file = "./data/adult.test"

print('Preparing data...')
# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)

org_linear_inputs = df_train[CATEGORICAL_COLUMNS]
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
linear_inputs = enc.fit_transform(org_linear_inputs).todense()

dnn_inputs = df_train[CONTINUOUS_COLUMNS]

y = df_train["income_bracket"].apply(lambda x: '>50K' in x).astype(int)

print(linear_inputs.shape, dnn_inputs.shape, y.shape)

print('Start building  model...')
epochs = 5

linear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(linear_inputs.shape[1],), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
linear_model.compile('adagrad', 'binary_crossentropy', metrics=['accuracy'])
linear_model.fit(linear_inputs, y, batch_size=32, epochs=epochs)

dnn_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=32),
        tf.keras.layers.Dense(units=8),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
dnn_model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(dnn_inputs, y, batch_size=32, epochs=epochs)

combined_model = tf.keras.experimental.WideDeepModel(linear_model, dnn_model, activation="relu")
combined_model.compile(optimizer=['sgd', 'adam'], loss='binary_crossentropy', metrics=['mse'])
combined_model.fit([linear_inputs, dnn_inputs], y, epochs)
