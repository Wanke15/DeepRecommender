import grpc

from grpc_service.wide_and_deep import wide_and_deep_pb2_grpc, wide_and_deep_pb2

channel = grpc.insecure_channel('localhost:50051')
stub = wide_and_deep_pb2_grpc.WideAndDeepGrpcStub(channel)

demo_data1 = {"workClass": "Private", "education": "11th", "maritalStatus": "Never-married",
                 "occupation": "Machine-op-inspct", "relationship": "Own-child", "race": "Black",
                 "gender": "Male", "nativeCountry": "United-States",
              "age": 25, "educationNum": 7, "capitalGain": 1, "capitalLoss": 1, "hoursPerWeek": 40}

adult = wide_and_deep_pb2.Adult(**demo_data1)
print(adult)


print("sending data...")
response = stub.Predict(adult)
print(response)
