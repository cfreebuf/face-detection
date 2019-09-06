# -*- coding: utf-8 -*-
# @file Makefile
# @author lidongming1@360.cn
# @date 2019-08-22 16:20
# @brief

ifeq ($(shell uname -m),x86_64)

# CC=/usr/local/bin/gcc
# CXX=/usr/local/bin/g++

# Modules
MODULE=faced

TF_ROOT=./tf

# INC_DIR += -I$(PYTHON_INC_DIR)

INC_DIR += -I/usr/local/include -I/usr/local/include/boost \
					 -I/usr/local/include/opencv4 \
					 -I/usr/local/include/gflags -I/usr/local/include/glog \
					 -I$(TF_ROOT)/include \
					 -Icommonlib -I. 

LIBS += -L/usr/local/lib -llmdb \
				-lboost_thread-mt \
				-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml \
				-lopencv_videoio -lopencv_imgcodecs \
				-lgflags -lglog -lgtest -lgrpc++ -lprotobuf \
				-Wl,-rpath,$(TF_ROOT)/lib -L$(TF_ROOT)/lib -ltensorflow

# STATIC_LIBS=Users/lidongming/tensorflow-1.14.0/tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a \
#						/Users/lidongming/tensorflow-1.14.0/tensorflow/contrib/makefile/gen/protobuf/lib/libprotobuf.a \
#						/Users/lidongming/tensorflow-1.14.0/tensorflow/contrib/makefile/downloads/nsync/builds/default.macos.c++11/libnsync.a

CXXFLAGS += -Wall -std=c++14 -O3 -Wno-deprecated-declarations -Wno-format
# -framework CoreFoundation

all: PRE_BUILD bin/$(MODULE) bin/unittest test/test_annoy
	@echo "[[32mBUILD[0m][Target:'[32mall[0m']"
	@echo "[[32mmake all done[0m]"

system-check:
	@echo "[[32mCHECK DEPENDENCY[0m]"

# 语法规范检查
style:
	python ../tools/cpplint.py --extensions=hpp,cpp --linelength=80 *.cpp

clean:
	@find . -name "*.o" | xargs -I {} rm {}
	@rm -rf bin/*
.phony:clean

PRE_BUILD:
	@mkdir -p bin
	@mkdir -p logs

IGNORE="main.cpp|test/main.cpp|test/test_annoy.cpp|back/*"
OBJS += $(patsubst %.cpp,%.o, $(shell find . -type f -name "*.cpp" | egrep -v $(IGNORE)))
OBJS += $(patsubst %.cc, %.o, $(shell find . -type f -name "*.cc"  | egrep -v $(IGNORE)))
OBJS += $(patsubst %.c,  %.o, $(shell find . -type f -name "*.c"   | egrep -v $(IGNORE)))

%.o:%.cpp
	@echo "[[32mBUILD[0m][Target:'[32m$<[0m']"
	$(CXX) $(CXXFLAGS) $(INC_DIR) -c $< -o $@

%.o:%.cc
	@echo "[[32mBUILD[0m][Target:'[32m$<[0m']"
	@$(CXX) $(CXXFLAGS) $(INC_DIR) -c $< -o $@

%.o:%.c
	@echo "[[32mBUILD[0m][Target:'[32m$<[0m']"
	@$(CXX) $(CXXFLAGS) $(INC_DIR) -c $< -o $@

bin/$(MODULE) : $(OBJS) 
	$(CXX) -o $@ $(INC_DIR) $(CXXFLAGS) main.cpp $(OBJS) $(LIBS)

bin/unittest : $(OBJS) 
	$(CXX) -o $@ $(INC_DIR) $(CXXFLAGS) test/main.cpp $(OBJS) $(LIBS)

test/test_annoy: $(OBJS) 
	$(CXX) -o $@ $(INC_DIR) $(CXXFLAGS) test/test_annoy.cpp $(OBJS) $(LIBS)

endif #ifeq ($(shell uname -m),x86_64)
