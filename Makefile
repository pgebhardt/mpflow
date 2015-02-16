# The Makefile for libmpflow
PROJECT := mpflow

# Directories
BUILD_DIR := build
PREFIX ?= /usr/local

# Target build architecture
ARM ?= 0
ifeq ($(ARM), 1)
	TARGET_ARCH := armv7-linux-gnueabihf
else
	TARGET_ARCH := x86_64-linux
endif
BUILD_DIR := $(BUILD_DIR)/$(TARGET_ARCH)

# The target shared library and static library names
LIB_BUILD_DIR := $(BUILD_DIR)/lib
NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).so
STATIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT)_static.a

# Cuda locations
CUDA_DIR ?= /usr/local/cuda
CUDA_INCLUDE_DIR := $(CUDA_DIR)/targets/$(TARGET_ARCH)/include
CUDA_LIB_DIR := $(CUDA_DIR)/targets/$(TARGET_ARCH)/lib

# Compiler
AR := ar rcs
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
INCLUDE_PATH := /usr/local/include ./include $(CUDA_INCLUDE_DIR)

# Compiler Flags
COMMON_FLAGS := $(addprefix -I, $(INCLUDE_PATH)) -DGIT_VERSION=\"$(GIT_VERSION)\" -O3
CFLAGS := -std=c++11 -fPIC
NVCCFLAGS := -Xcompiler -fpic -use_fast_math --ptxas-options=-v \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_32,code=sm_32 \
	-gencode=arch=compute_35,code=sm_35

# Target architecture specifiy compiler flags
ifeq ($(ARM), 1)
	NVCCFLAGS += -m32
endif

# Source Files
CXX_SRCS := $(shell find src -name "*.cpp")
HXX_SRCS := $(shell find include -name "*.h")
CU_SRCS := $(shell find src -name "*.cu")

# Object files
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
CU_OBJS := $(addprefix $(BUILD_DIR)/, ${CU_SRCS:.cu=.o})

# Build targets
.PHONY: all install clean

all: $(NAME) $(STATIC_NAME)

$(NAME): $(CXX_OBJS) $(CU_OBJS)
	@mkdir -p $(LIB_BUILD_DIR)
	$(CXX) -shared -o $@ $(CXX_OBJS) $(CU_OBJS)

$(STATIC_NAME): $(CXX_OBJS) $(CU_OBJS)
	@mkdir -p $(LIB_BUILD_DIR)
	$(AR) $@ $(CXX_OBJS) $(CU_OBJS)

$(BUILD_DIR)/%.o: %.cu $(HXX_SRCS)
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	$(NVCC) $(NVCCFLAGS) $(COMMON_FLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: %.cpp $(HXX_SRCS)
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	$(CXX) $(CFLAGS) $(COMMON_FLAGS) -c -o $@ $<

clean:
	@rm -rf $(BUILD_DIR)

install:
	install -m 0644 $(NAME) $(PREFIX)/lib
	install -m 0644 $(STATIC_NAME) $(PREFIX)/lib
	$(foreach f, $(HXX_SRCS), install -D -m 0644 $f $(PREFIX)/$f && ):
