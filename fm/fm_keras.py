import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

K = tf.keras.backend


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, embedding_dim, **kwargs):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        super(FMLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

    def build(self, input_shape):
        self.embedding = self.add_weight(name='embedding',
                                         shape=(self.input_dim, self.embedding_dim),
                                         initializer='glorot_uniform',
                                         trainable=True)
        super(FMLayer, self).build(input_shape)

    @tf.function
    def call(self, x):
        a = K.pow(K.dot(x, self.embedding), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.embedding, 2))
        return K.mean(a - b, 1, keepdims=True) * 0.5

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


def build_model(feature_dim, embedding_dim=8):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                  )(inputs)
    cross = FMLayer(feature_dim, embedding_dim)(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.optimizers.Adam(0.001),
                  metrics=[tf.metrics.AUC()])
    return model


if __name__ == '__main__':
    import pickle
    import numpy as np

    import tensorflow as tf
    import pandas as pd
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

    org_linear_inputs = df_train[CATEGORICAL_COLUMNS]

    enc = OneHotEncoder(handle_unknown='ignore')
    final_train_inputs = enc.fit_transform(org_linear_inputs).todense()

    # final_train_inputs = np.hstack((df_train[CONTINUOUS_COLUMNS].values, linear_inputs))

    # with open("./models/one_hot_encoder.b", 'wb') as f:
    #     pickle.dump(enc, f)

    # dnn_inputs = df_train[CONTINUOUS_COLUMNS]

    y = df_train["income_bracket"].apply(lambda x: '>50K' in x).astype(int)

    fm = build_model(final_train_inputs.shape[1], 15)
    # data = load_breast_cancer();data.data;data.target
    X_train, X_eval, y_train, y_eval = train_test_split(final_train_inputs, y, test_size=0.2,
                                                        random_state=27, stratify=y)
    fm.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_eval, y_eval))

    #############
    df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True)

    x_test_org = df_test[CATEGORICAL_COLUMNS]

    final_test_inputs = enc.transform(x_test_org).todense()
    # final_test_inputs = np.hstack((df_test[CONTINUOUS_COLUMNS].values, test_linear_inputs))

    y_test = df_test["income_bracket"].apply(lambda x: '>50K' in x).astype(int)

    fm.evaluate(final_test_inputs, y_test)

    # tf.keras.models.save_model(
    #     fm,
    #     './fm_keras_saved_model/1',
    #     overwrite=True,
    #     include_optimizer=True,
    #     save_format=None,
    #     signatures=None,
    #     options=None
    # )
