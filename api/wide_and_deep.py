import pickle

import numpy as np
import tensorflow as tf


class WideAndDeepModel:
    def __init__(self, model_json, model_weights, encoder, wide_cols, deep_cols, label_config):
        with open(model_json, "r") as json_file:
            self.model = tf.keras.models.model_from_json(json_file.read())
        self.model.load_weights(model_weights)

        with open(encoder, 'rb') as f:
            self.enc = pickle.load(f)

        self.wide_cols = wide_cols
        self.deep_cols = deep_cols

        self.label_config = label_config

    def predict(self, data):
        linear_inputs = np.array([[data.get(col) for col in self.wide_cols]])
        linear_inputs = self.enc.transform(linear_inputs)
        dnn_inputs = np.array([[data.get(col) for col in self.deep_cols]])
        pred_prob = self.model.predict([linear_inputs, dnn_inputs]).tolist()[0][0]
        pred_class = self.label_config.get(1) if pred_prob > self.label_config.get("threshold") else self.label_config.get(0)
        result = {"prob": pred_prob, "class": pred_class}
        return result

    def bacth_predict(self, data_list):
        linear_inputs = np.array([[data.get(col) for col in self.wide_cols] for data in data_list])
        linear_inputs = self.enc.transform(linear_inputs)
        dnn_inputs = np.array([[data.get(col) for col in self.deep_cols] for data in data_list])
        pred_porbs = [_[0] for _ in self.model.predict([linear_inputs, dnn_inputs]).tolist()]
        result = [{"prob": pred_prob, "class": self.label_config.get(1) if pred_prob > self.label_config.get(
            "threshold") else self.label_config.get(0)} for pred_prob in pred_porbs]
        return result


if __name__ == '__main__':
    LABEL_CONFIG = {1: ">50K", 0: "<=50K", "threshold": 0.5}
    CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                           "relationship", "race", "gender", "native_country"]
    CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                          "hours_per_week"]
    demo_data1 = {"workclass": "Private", "education": "11th", "marital_status": "Never-married",
                 "occupation": "Machine-op-inspct", "relationship": "Own-child", "race": "Black",
                 "gender": "Male", "native_country": "United-States",

                 "age": 25, "education_num": 7, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40}

    demo_data2 = {"workclass": "Local-gov", "education": "Assoc-acdm", "marital_status": "Married-civ-spouse",
                 "occupation": "Protective-serv", "relationship": "Husband", "race": "White",
                 "gender": "Male", "native_country": "United-States",

                 "age": 28, "education_num": 12, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40}

    base_dir = "../wide_and_deep/"
    model_instance = WideAndDeepModel(base_dir + "models/wide_and_deep/w_d_keras.json",
                                      base_dir + "models/wide_and_deep/w_d_keras",
                                      base_dir + "models/one_hot_encoder.b",
                                      CATEGORICAL_COLUMNS,
                                      CONTINUOUS_COLUMNS,
                                      LABEL_CONFIG)
    demo_result1 = model_instance.predict(demo_data1)
    print(demo_result1)

    demo_result2 = model_instance.predict(demo_data2)
    print(demo_result2)
