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

##############################
# Load build configuration
##############################
CONFIG_FILE ?= Makefile.config
include $(CONFIG_FILE)

##############################
# Main output directories
##############################
prefix ?= /usr/local
ROOT_BUILD_DIR := build

# adjust build dir for debug configuration
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(ROOT_BUILD_DIR)/debug
else
	BUILD_DIR := $(ROOT_BUILD_DIR)/release
endif

##############################
# Compiler
##############################
AR := ar rcs
CC ?= gcc
CXX ?= g++
NVCC := $(CUDA_TOOLKIT_ROOT)/bin/nvcc

ifdef CUDA_HOST_CXX
	NVCC += -ccbin=$(CUDA_HOST_CXX)
endif

# Target build architecture
TARGET_ARCH_NAME ?= $(shell $(CXX) -dumpmachine)
BUILD_DIR := $(BUILD_DIR)/$(TARGET_ARCH_NAME)

# get cuda directory for target architecture
CUDA_DIR := $(CUDA_TOOLKIT_ROOT)
ifdef CUDA_TARGET_DIR
	CUDA_DIR := $(CUDA_DIR)/targets/$(CUDA_TARGET_DIR)
endif

##############################
# The shared library and static library names
##############################
NAME := $(BUILD_DIR)/lib/lib$(PROJECT).so
STATIC_NAME := $(BUILD_DIR)/lib/lib$(PROJECT)_static.a

##############################
# Includes and libraries
##############################
LIBRARIES := distmesh_static qhull cudart_static cublas_static culibos pthread dl
LIBRARY_DIRS +=
INCLUDE_DIRS += $(CUDA_DIR)/include ./include ./utils/stringtools/include ./utils/json-parser ./utils/generics

# link aganinst librt, only if it exists
ifeq ($(shell echo "int main() {}" | $(CXX) -o /dev/null -x c - -lrt 2>&1),)
	LIBRARIES += rt
endif

# add (CUDA_DIR)/lib64 only if it exists
ifeq ("$(wildcard $(CUDA_DIR)/lib64)", "")
	LIBRARY_DIRS += $(CUDA_DIR)/lib
else
	LIBRARY_DIRS += $(CUDA_DIR)/lib64
endif

##############################
# Compiler Flags
##############################
GIT_VERSION := $(shell git describe --tags --long)
WARNINGS := -Wall -Wextra -Werror

COMMON_FLAGS := $(addprefix -I, $(INCLUDE_DIRS)) -DGIT_VERSION=\"$(GIT_VERSION)\"
CXXFLAGS := -std=c++11 -fPIC $(WARNINGS)
CFLAGS := -fPIC
NVCCFLAGS := -Xcompiler -fpic -use_fast_math $(CUDA_ARCH)

LINKFLAGS := -fPIC -static-libstdc++
LDFLAGS := $(addprefix -l, $(LIBRARIES)) $(addprefix -L, $(LIBRARY_DIRS)) $(addprefix -Xlinker -rpath , $(LIBRARY_DIRS))

# Set compiler flags for debug configuration
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -g -O0 -DDEBUG
	NVCCFLAGS += -G
else
	COMMON_FLAGS += -O3 -DNDEBUG
endif

##############################
# Source Files
##############################
CXX_SRCS := $(shell find src -name "*.cpp") utils/json-parser/json.c
HXX_SRCS := $(shell find include -name "*.h")
CU_SRCS := $(shell find src -name "*.cu")

# Object files
CXX_OBJS := $(addprefix $(BUILD_DIR)/objs/, $(CXX_SRCS:.c=.o))
CXX_OBJS := $(CXX_OBJS:.cpp=.o)
CU_OBJS := $(addprefix $(BUILD_DIR)/objs/, $(CU_SRCS:.cu=.o))

##############################
# Build targets
##############################
.PHONY: all install clean

all: $(NAME) $(STATIC_NAME)

$(NAME): $(CXX_OBJS) $(CU_OBJS)
	@echo [ Linking ] $@
	@mkdir -p $(BUILD_DIR)/lib
	@$(CXX) -shared -o $@ $(CXX_OBJS) $(CU_OBJS) $(COMMON_FLAGS) $(LDFLAGS) $(LINKFLAGS)

$(STATIC_NAME): $(CXX_OBJS) $(CU_OBJS)
	@echo [ Linking ] $@
	@mkdir -p $(BUILD_DIR)/lib
	@$(AR) $@ $(CXX_OBJS) $(CU_OBJS)

$(BUILD_DIR)/objs/%.o: %.cu $(HXX_SRCS)
	@echo [ NVCC ] $<
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	@$(NVCC) $(NVCCFLAGS) $(COMMON_FLAGS) -c -o $@ $<

$(BUILD_DIR)/objs/%.o: %.c
	@echo [ CC ] $<
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	@$(CC) $(CFLAGS) $(COMMON_FLAGS) -c -o $@ $<

$(BUILD_DIR)/objs/%.o: %.cpp $(HXX_SRCS)
	@echo [ CXX ] $<
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	@$(CXX) $(CXXFLAGS) $(COMMON_FLAGS) -c -o $@ $<

install: $(NAME) $(STATIC_NAME) $(HXX_SRCS) $(TOOLS_BINS)
	@install -m 0644 $(NAME) $(prefix)/lib
	@install -m 0644 $(STATIC_NAME) $(prefix)/lib
	@$(foreach f, $(HXX_SRCS), install -D -m 0644 $f $(prefix)/$f && ):
	@$(foreach f, $(TOOLS_BINS), install -m 0755 $f $(prefix)/bin && ):

clean:
	@rm -rf $(ROOT_BUILD_DIR)
