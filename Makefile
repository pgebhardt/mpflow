# fastEIT
#
# Copyright (C) 2012  Patrik Gebhardt
# Contact: patrik.gebhardt@rub.de

# Directories
SRC = src
INCLUDES = include
BUILD = build

# Install directories
INSTALL_INCLUDES = /usr/local/include/fasteit
INSTALL_LIB = /usr/local/lib

# Copmiler
CXX = clang++
NVCC = nvcc
CFLAGS = -std=c++11 -stdlib=libc++ -fPIC
NVFLAGS = -Xcompiler -fpic -m64 -arch=sm_30 --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -lineinfo
LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/local/lib -lcudart -lcublas

# Object files
_OBJ = matrix.o sparse_matrix.o mesh.o electrodes.o basis.o model.o conjugate.o sparse_conjugate.o forward.o inverse.o solver.o
OBJ = $(patsubst %, $(BUILD)/%, $(_OBJ))

# Cuda object files
_CUOBJ = matrix_kernel.cu_o sparse_matrix_kernel.cu_o model_kernel.cu_o conjugate_kernel.cu_o forward.cu_o
CUOBJ = $(patsubst %, $(BUILD)/%, $(_CUOBJ))

# Dependencies
_DEPS = fasteit.h dtype.h constants.h math.h matrix.h matrix_kernel.h sparse_matrix.h sparse_matrix_kernel.h mesh.h basis.h electrodes.h model.h model_kernel.h conjugate.h conjugate_kernel.h sparse_conjugate.h forward.h forward_cuda.h inverse.h solver.h
DEPS = $(patsubst %, $(INCLUDES)/%, $(_DEPS))

# Library
LIB = libfasteit.so

# Rule for library
$(LIB): $(OBJ) $(CUOBJ) $(DEPS)
	mkdir -p $(BUILD)
	$(CXX) -shared -o $(BUILD)/$(LIB) $(OBJ) $(CUOBJ) $(LDFLAGS)

# Rule for object files
$(BUILD)/%.o: $(SRC)/%.cpp $(DEPS)
	mkdir -p $(BUILD)
	$(CXX) $(CFLAGS) -c -o $@ $<

# Rule for cuda object files
$(BUILD)/%.cu_o: $(SRC)/%.cu $(DEPS)
	mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -c -o $@ $<

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
