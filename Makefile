# fastECT
#
# Copyright (C) 2012  Patrik Gebhardt
# Contact: patrik.gebhardt@rub.de

# os name
UNAME := $(shell uname)

# cuda paths
ifeq ($(UNAME), Linux)
CUDA_HOME = /usr/local/cuda
endif
ifeq ($(UNAME), Darwin)
CUDA_HOME = /usr/local/CUDA-5.0
endif

# Directories
SRC = src
INCLUDES = include
BUILD = build

# Install directories
INSTALL_INCLUDES = /usr/local/include/fastect
INSTALL_LIB = /usr/local/lib

# Copmiler
CC = clang
NVCC = $(CUDA_HOME)/bin/nvcc
CFLAGS = -fPIC
NVCFLAGS = -Xcompiler -fpic -m64
LDFLAGS = -L/usr/local/lib -llinalgcu -L$(CUDA_HOME)/lib64 -lcudart -lcublas

# Object files
_OBJ = mesh.o basis.o electrodes.o grid.o conjugate.o conjugate_sparse.o forward.o inverse.o solver.o
OBJ = $(patsubst %, $(BUILD)/%, $(_OBJ))

# Cuda object files
_CUOBJ = grid.cu_o conjugate.cu_o conjugate_sparse.cu_o forward.cu_o
CUOBJ = $(patsubst %, $(BUILD)/%, $(_CUOBJ))

# Dependencies
_DEPS = fastect.h mesh.h basis.h electrodes.h grid.h conjugate.h conjugate_sparse.h forward.h inverse.h solver.h
DEPS = $(patsubst %, $(INCLUDES)/%, $(_DEPS))

# Library
LIB = libfastect.so

# Rule for library
$(LIB): $(OBJ) $(CUOBJ) $(DEPS)
	mkdir -p $(BUILD)
	$(CC) -shared -o $(BUILD)/$(LIB) $(OBJ) $(CUOBJ) $(LDFLAGS)

# Rule for object files
$(BUILD)/%.o: $(SRC)/%.c $(DEPS)
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

# Rule for cuda object files
$(BUILD)/%.cu_o: $(SRC)/%.cu $(DEPS)
	mkdir -p $(BUILD)
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

# Install
install:
	mkdir -p $(INSTALL_INCLUDES)
	install -m 0644 $(INCLUDES)/*.h $(INSTALL_INCLUDES)
	install -m 0644 $(BUILD)/$(LIB) $(INSTALL_LIB)

# Uninstall
uninstall:
	rm -rf $(INSTALL_INCLUDES)
	rm -rf $(INSTALL_LIB)/$(LIB)

# Cleanup
clean:
	rm -rf $(BUILD)

# Flags
.PHONY: clean install
