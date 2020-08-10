from flask import Flask, jsonify

from api.utils import get_body
from api.wide_and_deep import WideAndDeepModel

app = Flask("DeepRecommender")
LABEL_CONFIG = {1: ">50K", 0: "<=50K", "threshold": 0.4}
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]
base_dir = "./wide_and_deep/"
wide_and_deep_model = WideAndDeepModel(base_dir + "models/wide_and_deep/w_d_keras.json",
                                       base_dir + "models/wide_and_deep/w_d_keras",
                                       base_dir + "models/one_hot_encoder.b",
                                       CATEGORICAL_COLUMNS,
                                       CONTINUOUS_COLUMNS,
                                       LABEL_CONFIG)


@app.route("/wide-and-deep/adult", methods=['POST'])
def wide_and_deep_inference():
    data = get_body()
    try:
        result = wide_and_deep_model.predict(data)
        response = {"data": result, "msg": "success", "code": 200}
    except Exception as e:
        response = {"data": [], "msg": "{}".format(e), "code": 500}
    return jsonify(response)


@app.route("/wide-and-deep/adult-batch", methods=['POST'])
def wide_and_deep_inference_batch():
    data = get_body()
    try:
        result = wide_and_deep_model.bacth_predict(data)
        response = {"data": result, "msg": "success", "code": 200}
    except Exception as e:
        response = {"data": [], "msg": "{}".format(e), "code": 500}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3723)
