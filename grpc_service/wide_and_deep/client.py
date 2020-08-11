import grpc

from grpc_service.wide_and_deep import wide_and_deep_pb2_grpc, wide_and_deep_pb2

channel = grpc.insecure_channel('localhost:50051')
stub = wide_and_deep_pb2_grpc.WideAndDeepGrpcStub(channel)

demo_data1 = {"workclass": "Private", "education": "11th", "marital_status": "Never-married",
                 "occupation": "Machine-op-inspct", "relationship": "Own-child", "race": "Black",
                 "gender": "Male", "native_country": "United-States",
              "age": 25, "education_num": 7, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40}

demo_data2 = {"workclass": "Local-gov", "education": "Assoc-acdm", "marital_status": "Married-civ-spouse",
                 "occupation": "Protective-serv", "relationship": "Husband", "race": "White",
                 "gender": "Male", "native_country": "United-States",

                 "age": 28, "education_num": 12, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40}


print("sending data...")
response = stub.Predict(wide_and_deep_pb2.Adult(**demo_data1))
print(response)

response = stub.Predict(wide_and_deep_pb2.Adult(**demo_data2))
print(response)
