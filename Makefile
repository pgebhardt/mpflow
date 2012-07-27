# fastECT
#
# Copyright (C) 2012  Patrik Gebhardt
# Contact: patrik.gebhardt@rub.de
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# cuda paths
CUDA_HOME = /usr/local/cuda

# Copmiler
CC = clang
NVCC = $(CUDA_HOME)/bin/nvcc
CFLAGS =
NVCFLAGS =

LDFLAGS = -L/usr/local/lib -llinalgcu -L/usr/local/cuda/lib64 -lcudart -lcublas -lm

# Directories
SRC = src
BUILD = build

# Object files
_OBJ = mesh.o basis.o electrodes.o grid.o image.o conjugate.o forward.o solver.o
OBJ = $(patsubst %, $(BUILD)/%, $(_OBJ))

# Cuda object files
_CUOBJ = grid.cu_o image.cu_o conjugate.cu_o
CUOBJ = $(patsubst %, $(BUILD)/%, $(_CUOBJ))

# Dependencies
_DEPS = fastect.h mesh.h basis.h electrodes.h grid.h image.h conjugate.h forward.h solver.h
DEPS = $(patsubst %, $(SRC)/%, $(_DEPS))

# Output file
BIN = fastECT
FORWARD_SOLVER = fastECT_forward

# Rule for library
$(BIN): $(BUILD)/main.o $(OBJ) $(CUOBJ) $(DEPS)
	$(CC) $(CFLAGS) -o $(BIN) $(BUILD)/main.o $(OBJ) $(CUOBJ) $(LDFLAGS)

$(FORWARD_SOLVER): $(BUILD)/forward_solver.o $(OBJ) $(CUOBJ) $(DEPS)
	$(CC) $(CFLAGS) -o $(FORWARD_SOLVER) $(BUILD)/forward_solver.o $(OBJ) $(CUOBJ) $(LDFLAGS)

# Rule for object files
$(BUILD)/%.o: $(SRC)/%.c $(DEPS)
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

# Rule for cuda object files
$(BUILD)/%.cu_o: $(SRC)/%.cu $(DEPS)
	mkdir -p $(BUILD)
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

# Cleanup
clean:
	rm -rf $(BUILD)

# Flags
.PHONY: clean
