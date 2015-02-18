# --------------------------------------------------------------------
# This file is part of mpFlow.
#
# mpFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# mpFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mpFlow. If not, see <http:#www.gnu.org/licenses/>.
#
# Copyright (C) 2015 Patrik Gebhardt
# Contact: patrik.gebhardt@rub.de
# --------------------------------------------------------------------

# The Makefile for libmpflow
PROJECT := mpflow

# Directories
BUILD_DIR := build
prefix ?= /usr/local

# Cuda locations
CUDA_TOOLKIT_ROOT ?= /usr/local/cuda
CUDA_DIR := $(CUDA_TOOLKIT_ROOT)

# Compiler
AR := ar rcs
NVCC := $(CUDA_TOOLKIT_ROOT)/bin/nvcc
CXX := clang++

# Cross compile for arm architecture
ARM ?= 0
ifeq ($(ARM), 1)
	CXX := arm-linux-gnueabihf-g++
	TARGET_ARCH := armv7-linux-gnueabihf
	CUDA_DIR := $(CUDA_TOOLKIT_ROOT)/targets/$(TARGET_ARCH)
endif

# Target build architecture
TARGET_ARCH ?= $(shell uname -m)-$(shell uname)
BUILD_DIR := $(BUILD_DIR)/$(TARGET_ARCH)

# The target shared library and static library names
LIB_BUILD_DIR := $(BUILD_DIR)/lib
NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).so
STATIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT)_static.a

# Version Define
GIT_VERSION := $(shell git describe --tags --long)

# Includes and libraries
LIBRARIES := cudart_static cublas_static distmesh_static qhull culibos pthread dl rt
LIBRARY_DIRS := $(CUDA_DIR)/lib
INCLUDE_DIRS := $(CUDA_DIR)/include /usr/local/include ./include

# add <cuda>/lib64 only if it exists
ifneq ("$(wildcard $(CUDA_DIR)/lib64)", "")
	LIBRARY_DIRS += $(CUDA_DIR)/lib64
endif

# Compiler Flags
COMMON_FLAGS := $(addprefix -I, $(INCLUDE_DIRS)) -DGIT_VERSION=\"$(GIT_VERSION)\" -O3
CFLAGS := -std=c++11 -fPIC
NVCCFLAGS := -Xcompiler -fpic -use_fast_math --ptxas-options=-v \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_32,code=sm_32 \
	-gencode=arch=compute_35,code=sm_35
LINKFLAGS := -O3 -fPIC -static-libgcc -static-libstdc++
LDFLAGS := $(addprefix -l, $(LIBRARIES)) $(addprefix -L, $(LIBRARY_DIRS))

# Tell nvcc how to build for arm architecture
ifeq ($(ARM), 1)
	NVCCFLAGS += -m32 -ccbin=$(CXX)
endif

# Source Files
CXX_SRCS := $(shell find src -name "*.cpp")
HXX_SRCS := $(shell find include -name "*.h")
CU_SRCS := $(shell find src -name "*.cu")
TOOL_SRCS := $(shell find tools -name "*.cpp")

# Object files
CXX_OBJS := $(addprefix $(BUILD_DIR)/objs/, $(CXX_SRCS:.cpp=.o))
CU_OBJS := $(addprefix $(BUILD_DIR)/objs/, $(CU_SRCS:.cu=.o))
TOOL_OBJS := $(addprefix $(BUILD_DIR)/objs/, $(TOOL_SRCS:.cpp=.o))
TOOL_BINS := $(patsubst tools%.cpp, $(BUILD_DIR)/bin%, $(TOOL_SRCS))

# Build targets
.PHONY: all install clean tools

all: $(NAME) $(STATIC_NAME) tools

tools: $(TOOL_BINS)

$(TOOL_BINS): $(BUILD_DIR)/bin/% : $(BUILD_DIR)/objs/tools/%.o $(STATIC_NAME)
	@mkdir -p $(BUILD_DIR)/bin
	$(CXX) $< $(STATIC_NAME) -o $@ $(LDFLAGS) $(LINKFLAGS)

$(NAME): $(CXX_OBJS) $(CU_OBJS)
	@mkdir -p $(LIB_BUILD_DIR)
	$(CXX) -shared -o $@ $(CXX_OBJS) $(CU_OBJS) $(LDFLAGS) $(LINKFLAGS)

$(STATIC_NAME): $(CXX_OBJS) $(CU_OBJS)
	@mkdir -p $(LIB_BUILD_DIR)
	$(AR) $@ $(CXX_OBJS) $(CU_OBJS)

$(BUILD_DIR)/objs/%.o: %.cu $(HXX_SRCS)
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	$(NVCC) $(NVCCFLAGS) $(COMMON_FLAGS) -c -o $@ $<

$(BUILD_DIR)/objs/%.o: %.cpp $(HXX_SRCS)
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	$(CXX) $(CFLAGS) $(COMMON_FLAGS) -c -o $@ $<

install: $(NAME) $(STATIC_NAME) $(HXX_SRCS) $(TOOL_BINS)
	@install -m 0644 $(NAME) $(prefix)/lib
	@install -m 0644 $(STATIC_NAME) $(prefix)/lib
	@$(foreach f, $(HXX_SRCS), install -D -m 0644 $f $(prefix)/$f && ):
	@$(foreach f, $(TOOL_BINS), install -m 0755 $f $(prefix)/bin && ):

clean:
	@rm -rf $(BUILD_DIR)
