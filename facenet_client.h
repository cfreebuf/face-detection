// File   facenet_client.h
// Author lidongming1@360.cn
// Date   2019-09-05 15:26:41
// Brief

#ifndef _FACENETC_CLIENT_H_
#define _FACENETC_CLIENT_H_

#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <grpcpp/grpcpp.h>
#include "protos/facenet.grpc.pb.h"
#include "common/common_gflags.h"
 
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using facenet_server::EmbeddingRequest;
using facenet_server::EmbeddingReply;
using facenet_server::FaceNet;
 
class FaceNetClient {
 public:
  FaceNetClient(std::shared_ptr<Channel> channel)
      : stub_(FaceNet::NewStub(channel)) {}
 
  EmbeddingReply GenerateEmbedding(EmbeddingRequest& request) {
    EmbeddingReply reply;
 
    ClientContext context;
    std::chrono::system_clock::time_point deadline = 
      std::chrono::system_clock::now() 
      + std::chrono::milliseconds(FLAGS_facenet_client_timeout);
    context.set_deadline(deadline);

    Status status = stub_->GenerateEmbedding(&context, request, &reply);
 
    if (status.ok()) {
      reply.set_error(0);
    } else {
      reply.set_error(status.error_code());
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
    return reply;
  }
 
 private:
  std::unique_ptr<FaceNet::Stub> stub_;
};
 
#if 0
int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  FaceNetClient greeter(grpc::CreateChannel(
      "127.0.0.1:50051", grpc::InsecureChannelCredentials()));
  std::string user("world");
  std::string reply = greeter.SayHello(user);
  std::cout << "Greeter received: " << reply << std::endl;
 
  return 0;
}
#endif

#endif
