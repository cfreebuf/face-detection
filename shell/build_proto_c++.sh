protoc --grpc_out=protos --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` facenet.proto
protoc -Iprotos --cpp_out=protos facenet.proto
