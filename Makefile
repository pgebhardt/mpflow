# The Makefile for libmpflow
PROJECT := mpflow

# Directories
BUILD_DIR := build
PREFIX ?= /usr/local

# Target build architecture
ARM ?= 0
ifeq ($(ARM), 1)
	TARGET_ARCH := arm-linux-gnueabihf
else
	TARGET_ARCH := x86_64-linux
endif
BUILD_DIR := $(BUILD_DIR)/$(TARGET_ARCH)

# Cuda locations
CUDA_DIR := /usr/local/cuda-6.5
CUDA_LIB_DIR := $(CUDA_DIR)/targets/$(TARGET_ARCH)/lib
CUDA_INCLUDE_DIR := $(CUDA_DIR)/targets/$(TARGET_ARCH)/include

# Compiler
ifeq ($(ARM), 1)
	CXX := arm-linux-gnueabihf-g++
	NVCC := $(CUDA_DIR)/bin/nvcc -ccbin $(CXX)
else
	CXX := clang++
	NVCC := $(CUDA_DIR)/bin/nvcc
endif

# Version Define
GIT_VERSION := $(shell git describe --tags --long)

# Includes and libraries
LD_PATH := /usr/local/lib $(CUDA_LIB_DIR)
INCLUDE_PATH := /usr/local/include ./include $(CUDA_INCLUDE_DIR)

# Compiler Flags
COMMON_FLAGS := $(addprefix -I, $(INCLUDE_PATH)) -DGIT_VERSION=\"$(GIT_VERSION)\" -O3
CFLAGS := -std=c++11 -fPIC
LINKER_FLAGS := $(addprefix -L, $(LD_PATH))
NVCCFLAGS := -Xcompiler -fpic -use_fast_math --ptxas-options=-v

# Target architecture specifiy compiler flags
ifeq ($(ARM), 1)
	NVCCFLAGS += -target-cpu-arch=ARM -m32 -Xptxas '-dlcm=ca' -target-os-variant=Linux \
		-gencode=arch=compute_32,code=sm_32
else
	NVCCFLAGS += 	-gencode=arch=compute_30,code=sm_30 \
					-gencode=arch=compute_35,code=sm_35
endif

# Source Files
CXX_SRCS := $(shell find src -name "*.cpp")
HXX_SRCS := $(shell find include -name "*.h")
CU_SRCS := $(shell find src -name "*.cu")

# Object files
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
CU_OBJS := $(addprefix $(BUILD_DIR)/, ${CU_SRCS:.cu=.o})

# Build targets
.PHONY: install clean

lib$(PROJECT).so: $(CXX_OBJS) $(CU_OBJS)
	$(CXX) -shared -o $(BUILD_DIR)/$@ $(CXX_OBJS) $(CU_OBJS) $(LINKER_FLAGS)

$(BUILD_DIR)/%.o: %.cu $(HXX_SRCS)
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	$(NVCC) $(NVCCFLAGS) $(COMMON_FLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: %.cpp $(HXX_SRCS)
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	$(CXX) $(CFLAGS) $(COMMON_FLAGS) -c -o $@ $<

clean:
	@rm -rf $(BUILD_DIR)

install:
	install -m 0644 $(BUILD_DIR)/lib$(PROJECT).so $(PREFIX)/lib
	$(foreach f, $(HXX_SRCS), install -D -m 0644 $f $(PREFIX)/$f && ):
