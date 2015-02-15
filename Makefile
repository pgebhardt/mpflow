# The Makefile for libmpflow
PROJECT := mpflow

# Directories
SOURCE_DIR := src
BUILD_DIR := build
CUDA_DIR := /usr/local/cuda

# Compiler
CXX := arm-linux-gnueabihf-g++
NVCC := $(CUDA_DIR)/bin/nvcc -ccbin $(CXX)
LD := $(NVCC)

# Version Define
GIT_VERSION := $(shell git describe --tags --long)

# Include paths
INCLUDES := /usr/local/include ./include $(CUDA_DIR)/targets/armv7-linux-gnueabihf/include

# Compiler Flags
CFLAGS := $(addprefix -I, $(INCLUDES)) -DGIT_VERSION=\"$(GIT_VERSION)\" -O3
NVCCFLAGS := -Xcompiler -fpic -use_fast_math --ptxas-options=-v \
	-target-cpu-arch=ARM -m32 -Xptxas '-dlcm=ca' -target-os-variant=Linux \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_32,code=sm_32 \
	-gencode=arch=compute_35,code=sm_35

# Source Files
CXX_SRCS := $(shell find $(SOURCE_DIR) -name "*.cpp")
CU_SRCS := $(shell find $(SOURCE_DIR) -name "*.cu")

# Object files
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
CU_OBJS := $(addprefix $(BUILD_DIR)/, ${CU_SRCS:.cu=.o})

lib$(PROJECT).a: $(CXX_OBJS) $(CU_OBJS)
	$(LD) -lib -o $(BUILD_DIR)/$@ $(CXX_OBJS) $(CU_OBJS)

$(BUILD_DIR)/%.o: %.cu
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(EXTRA_CFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: %.cpp
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	$(CXX) -std=gnu++11 $(CFLAGS) $(EXTRA_CFLAGS) -c -o $@ $<

clean:
	@rm -rf $(BUILD_DIR)
