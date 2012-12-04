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
_OBJ = mesh.o electrodes.o linearBasis.o model.o conjugate.o #sparseConjugate.o forward.o inverse.o solver.o
OBJ = $(patsubst %, $(BUILD)/%, $(_OBJ))

# Cuda object files
_CUOBJ = matrix.cu_o sparse.cu_o model.cu_o conjugate.cu_o #forward.cu_o
CUOBJ = $(patsubst %, $(BUILD)/%, $(_CUOBJ))

# Dependencies
_DEPS = fasteit.hpp dtype.hpp math.hpp matrix.hpp sparse.hpp mesh.hpp basis.hpp electrodes.hpp model.hpp model.hcu conjugate.hpp sparseConjugate.hpp forward.hpp inverse.hpp solver.hpp
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
	install -m 0644 $(INCLUDES)/*.hpp $(INSTALL_INCLUDES)
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
