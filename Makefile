# fastECT
#
# Copyright (C) 2012  Patrik Gebhardt
# Contact: patrik.gebhardt@rub.de

# cuda paths
CUDA_HOME = /usr/local/cuda

# Directories
SRC = src
INCLUDES = include
BUILD = build
EXAMPLES = examples
BIN = bin

# Install directories
INSTALL_INCLUDES = /usr/local/include/fastect
INSTALL_LIB = /usr/local/lib

# Copmiler
CC = clang
NVCC = $(CUDA_HOME)/bin/nvcc
CFLAGS =
NVCFLAGS = -m64
LDFLAGS = -L$(BUILD) -lfastect -L$(INSTALL_LIB) -llinalgcu -L$(CUDA_HOME)/lib -lcudart -lcublas -lconfig -lm

# Object files
_OBJ = mesh.o basis.o electrodes.o grid.o image.o conjugate.o forward.o inverse.o solver.o
OBJ = $(patsubst %, $(BUILD)/%, $(_OBJ))

# Cuda object files
_CUOBJ = grid.cu_o image.cu_o conjugate.cu_o solver.cu_o
CUOBJ = $(patsubst %, $(BUILD)/%, $(_CUOBJ))

# Dependencies
_DEPS = fastect.h mesh.h basis.h electrodes.h grid.h image.h conjugate.h forward.h inverse.h solver.h
DEPS = $(patsubst %, $(INCLUDES)/%, $(_DEPS))

# Examples
EXAMPLEOBJ = $(patsubst $(EXAMPLES)/%.c, %, $(wildcard $(EXAMPLES)/*.c))

# Library
LIB = libfastect.a

# Rule for library
$(LIB): $(OBJ) $(CUOBJ) $(DEPS)
	mkdir -p $(BUILD)
	ar rc $(BUILD)/$(LIB) $(OBJ) $(CUOBJ)
	ranlib $(BUILD)/$(LIB)

# Rule for examples
examples: $(LIB) $(EXAMPLEOBJ)
	mkdir -p $(BIN)/output
	cp -r $(EXAMPLES)/input $(BIN)/
	cp -r $(EXAMPLES)/script.py $(BIN)/

# Rule for object files
$(BUILD)/%.o: $(SRC)/%.c $(DEPS)
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

# Rule for cuda object files
$(BUILD)/%.cu_o: $(SRC)/%.cu $(DEPS)
	mkdir -p $(BUILD)
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

# Rule for example executables
%: $(EXAMPLES)/%.c
	mkdir -p $(BIN)
	$(CC) $(CFLAGS) -o $(BIN)/$@ $< $(LDFLAGS)

# Install
install:
	mkdir -p $(INSTALL_INCLUDES)
	install -m 0644 $(INCLUDES)/*.h $(INSTALL_INCLUDES)
	install -m 0644 $(BUILD)/$(LIB) $(INSTALL_LIB)

# Uninstall
uninstall:
	rm -rf $(INCLUDES)
	rm -rf $(LIBS)/$(LIB)

# Cleanup
clean:
	rm -rf $(BUILD)
	rm -rf $(BIN)

# Flags
.PHONY: clean install
