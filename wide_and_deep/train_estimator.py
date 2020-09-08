import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

from sklearn.metrics import roc_auc_score

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

# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(train_file, names=COLUMNS)
df_test = pd.read_csv(test_file, names=COLUMNS)
df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

model_dir = "./models/w_n_d/"
m = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=CATEGORICAL_COLUMNS,
    dnn_feature_columns=CONTINUOUS_COLUMNS,
    dnn_hidden_units=[100, 50])

_HASH_BUCKET_SIZE = 1000


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        # 全是离散特征
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    # 交叉特征
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'],
            hash_bucket_size=_HASH_BUCKET_SIZE),
    ]

    # wide特征列
    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        # tf.feature_column.indicator_column(workclass),
        # tf.feature_column.indicator_column(education),
        # tf.feature_column.indicator_column(marital_status),
        # tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        # tf.feature_column.embedding_column(occupation, dimension=8),

        tf.feature_column.embedding_column(workclass, dimension=8),
        tf.feature_column.embedding_column(education, dimension=8),
        tf.feature_column.embedding_column(marital_status, dimension=8),
        tf.feature_column.embedding_column(relationship, dimension=8),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['<=50K']]

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def input_fn(data_path, shuffle, num_epochs, batch_size):
    def parse_csv(value):
        # columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        columns = tf.io.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        if labels is not None:
            classes = tf.cast(x=tf.equal(labels, '>50K'), dtype=tf.int32)  # binary classification
            return features, classes
        else:
            return features

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_path)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def predict_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.feature_column.numeric_column(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.feature_column.categorical_column_with_identity(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items())
    feature_cols.update(dict(categorical_cols.items()))
    # Converts the label column into a constant Tensor.
    # Returns the feature columns and the label.
    return feature_cols


def run():
    ROOT_PATH = './data/'
    TRAIN_PATH = ROOT_PATH + 'adult.train'
    EVAL_PATH = ROOT_PATH + 'adult.test'
    PREDICT_PATH = ROOT_PATH + 'adult.test'
    MODEL_PATH = './models/adult_model'
    EXPORT_PATH = './models/adult_model_export'
    wide_columns, deep_columns = build_model_columns()

    # os.system('rm -rf {}'.format(MODEL_PATH))
    config = tf.estimator.RunConfig(save_checkpoints_steps=100)
    estimator = tf.estimator.DNNLinearCombinedClassifier(model_dir=MODEL_PATH,
                                                         linear_feature_columns=wide_columns,
                                                         # linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01),
                                                         linear_optimizer=tf.optimizers.Ftrl(learning_rate=0.001),
                                                         dnn_feature_columns=deep_columns,
                                                         dnn_hidden_units=[256, 64, 16],
                                                         # dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                         dnn_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                                                         config=config)

    # def my_auc(labels, predictions):
    #     auc_metric = tf.keras.metrics.AUC(name="my_auc")
    #     auc_metric.update_state(y_true=labels, y_pred=predictions['probabilities'][:, 1])
    #     return {'auc': auc_metric}
    #
    # estimator = tf.estimator.add_metrics(estimator, my_auc)

    # Train the model.
    estimator.train(
        input_fn=lambda: input_fn(data_path=TRAIN_PATH, shuffle=True, num_epochs=100, batch_size=64), steps=2000)
    """
    steps: 最大训练次数，模型训练次数由训练样本数量、num_epochs、batch_size共同决定，通过steps可以提前停止训练
    """
    # Evaluate the model.
    # eval_result = estimator.evaluate(
    #     input_fn=lambda: input_fn(data_path=TRAIN_PATH, shuffle=False, num_epochs=1, batch_size=40))
    #
    # print('Train set accuracy:', eval_result)

    eval_result = estimator.evaluate(
        input_fn=lambda: input_fn(data_path=EVAL_PATH, shuffle=False, num_epochs=1, batch_size=40))

    print('Test set accuracy:', eval_result)

    # Predict.
    # pred_dict = estimator.predict(
    #     input_fn=lambda: input_fn(data_path=PREDICT_PATH, shuffle=False, num_epochs=1, batch_size=40))
    #
    # y_test = df_test["income_bracket"].apply(lambda x: '>50K' in x).astype(int)
    # print('Test AUC: ')
    # print(roc_auc_score(y_test, [pred_res['probabilities'][1] for pred_res in pred_dict]))

    columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_saved_model(EXPORT_PATH, serving_input_fn)


if __name__ == '__main__':
    run()