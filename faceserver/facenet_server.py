# -*- coding: utf-8 -*-
# @file facenet_server.py
# @author lidongming1@360.cn
# @date 2019-09-06 13:11
# @brief

from concurrent import futures
import time
import grpc
import facenet_pb2
import facenet_pb2_grpc
from face_encoder import Encoder

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
encoder = Encoder()

class FaceNet(facenet_pb2_grpc.FaceNetServicer):
    def GenerateEmbedding(self, request, context):
        time_start=time.time()
        res = facenet_pb2.EmbeddingReply()
        face_dims = encoder.generate_embedding_from_base64(request.image_base64)
        if not len(face_dims):
            res.error = -1
        else:
            res.dim.extend(face_dims)
        time_end=time.time()
        print 'process request, time cost: %dms' % ((time_end - time_start) * 1000)
        return res

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    facenet_pb2_grpc.add_FaceNetServicer_to_server(FaceNet(), server)
    server.add_insecure_port('[::]:10101')
    server.start()
    #  server.wait_for_termination()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
