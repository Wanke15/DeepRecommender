from concurrent.futures.thread import ThreadPoolExecutor

import grpc

from grpc_service.wide_and_deep import wide_and_deep_pb2_grpc


def serve():
    LABEL_CONFIG = {1: ">50K", 0: "<=50K", "threshold": 0.4}
    CATEGORICAL_COLUMNS = ["workclass", "education", "maritalStatus", "occupation",
                           "relationship", "race", "gender", "nativeCountry"]
    CONTINUOUS_COLUMNS = ["age", "educationNum", "capitalGain", "capitalLoss",
                          "hoursPerWeek"]
    base_dir = "/Users/jeff/PycharmProjects/DeepRecommender/wide_and_deep/"

    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    wide_and_deep_pb2_grpc.add_WideAndDeepGrpcServicer_to_server(
        wide_and_deep_pb2_grpc.WideAndDeepGrpcServicer(base_dir + "models/wide_and_deep/w_d_keras.json",
                                                       base_dir + "models/wide_and_deep/w_d_keras",
                                                       base_dir + "models/one_hot_encoder.b",
                                                       CATEGORICAL_COLUMNS,
                                                       CONTINUOUS_COLUMNS,
                                                       LABEL_CONFIG), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
