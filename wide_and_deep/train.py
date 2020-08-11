import pickle

import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_auc_score
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

train_file = "./data/adult.train"
test_file = "./data/adult.test"

print('Preparing data...')
# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)

org_linear_inputs = df_train[CATEGORICAL_COLUMNS]

enc = OneHotEncoder(handle_unknown='ignore')
linear_inputs = enc.fit_transform(org_linear_inputs).todense()
with open("./models/one_hot_encoder.b", 'wb') as f:
    pickle.dump(enc, f)

dnn_inputs = df_train[CONTINUOUS_COLUMNS]

y = df_train["income_bracket"].apply(lambda x: '>50K' in x).astype(int)

print(linear_inputs.shape, dnn_inputs.shape, y.shape)

print('Start building  model...')
epochs = 10

linear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(linear_inputs.shape[1], ), name="linear_feature_layer", activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
linear_model.compile('adagrad', 'binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
# linear_model.fit(linear_inputs, y, batch_size=32, epochs=epochs)

auc_score = roc_auc_score(y, linear_model.predict(linear_inputs))
print("Pure wide auc: ", auc_score)

dnn_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=32),
        tf.keras.layers.Dense(units=8, name="dnn_feature_layer"),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
dnn_model.compile('rmsprop', 'binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
# dnn_model.fit(dnn_inputs, y, batch_size=32, epochs=epochs)
auc_score = roc_auc_score(y, dnn_model.predict(dnn_inputs))
print("Pure deep auc: ", auc_score)


combined_features = tf.keras.layers.concatenate([linear_model.get_layer("linear_feature_layer").output,
                                                 dnn_model.get_layer("dnn_feature_layer").output])
outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="combined_output")(combined_features)

combined_model = tf.keras.models.Model(inputs=[*linear_model.inputs, *dnn_model.inputs], outputs=outputs)
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision()])
# combined_model.summary()

combined_model.fit([linear_inputs, dnn_inputs], y, batch_size=64, epochs=epochs)

y_pred = combined_model.predict([linear_inputs, dnn_inputs])

from sklearn.metrics import roc_auc_score

auc_score = roc_auc_score(y, y_pred)
print("Pure w&d auc: ", auc_score)

print("Saving model...")
model_json = combined_model.to_json(indent=4)
with open("./models/wide_and_deep/w_d_keras.json", "w") as json_file:
    json_file.write(model_json)
combined_model.save_weights("./models/wide_and_deep/w_d_keras")
