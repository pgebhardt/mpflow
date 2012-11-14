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
CC ?= clang
NVCC = nvcc
CFLAGS = -fPIC
NVFLAGS = -Xcompiler -fpic -m64 -arch=sm_30 --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -lineinfo
LDFLAGS = -L/usr/local/lib -llinalgcu -lcudart -lcublas

# Object files
_OBJ = mesh.o basis.o electrodes.o model.o conjugate.o sparse_conjugate.o forward.o inverse.o solver.o
OBJ = $(patsubst %, $(BUILD)/%, $(_OBJ))

# Cuda object files
_CUOBJ = model.cu_o conjugate.cu_o forward.cu_o
CUOBJ = $(patsubst %, $(BUILD)/%, $(_CUOBJ))

# Dependencies
_DEPS = fasteit.h mesh.h basis.h electrodes.h model.h conjugate.h sparse_conjugate.h forward.h inverse.h solver.h
DEPS = $(patsubst %, $(INCLUDES)/%, $(_DEPS))

# Library
LIB = libfasteit.so

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
