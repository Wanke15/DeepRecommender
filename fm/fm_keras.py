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
    fm = build_model(30, 15)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    fm.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    tf.keras.models.save_model(
        fm,
        './fm_keras_saved_model/1',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
