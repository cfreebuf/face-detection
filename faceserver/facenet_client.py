from __future__ import print_function

import grpc
import time
import facenet_pb2
import facenet_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = facenet_pb2_grpc.FaceNetStub(channel)

    for i in range(100):
        with open("b.txt") as f:
            line = f.readline()
            time_start=time.time()
            response = stub.GenerateEmbedding(facenet_pb2.EmbeddingRequest(image_base64=line))
            time_end=time.time()
            duration = (time_end - time_start) * 1000
            #  print(response.dim)
            print('time cost:', duration, 'ms')

if __name__ == '__main__':
    run()
