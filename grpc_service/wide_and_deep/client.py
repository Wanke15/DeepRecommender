import grpc

from grpc_service.wide_and_deep import wide_and_deep_pb2_grpc, wide_and_deep_pb2

channel = grpc.insecure_channel('localhost:50051')
stub = wide_and_deep_pb2_grpc.WideAndDeepGrpcStub(channel)

demo_data1 = {"workclass": "Private", "education": "11th", "marital_status": "Never-married",
                 "occupation": "Machine-op-inspct", "relationship": "Own-child", "race": "Black",
                 "gender": "Male", "native_country": "United-States",
              "age": 25, "education_num": 7, "capital_gain": 1, "capital_loss": 1, "hours_per_week": 40}

adult = wide_and_deep_pb2.Adult(**demo_data1)
print(adult)


print("sending data...")
response = stub.Predict(adult)
print(response)
