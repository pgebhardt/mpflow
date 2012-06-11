# ert
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

# Copmiler and flags
CC = clang
CFLAGS = -fblocks
LDFLAGS = -lBlocksRuntime -lactor -llinalg -ldispatch

# Directories
SRC = src
BUILD = build

# Object files
_OBJ = main.o mesh.o
OBJ = $(patsubst %, $(BUILD)/%, $(_OBJ))

# Dependencies
_DEPS = mesh.h
DEPS = $(patsubst %, $(SRC)/%, $(_DEPS))

# Output file
BIN = ert

# Rule for library
$(BIN): $(OBJ) $(DEPS)
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BIN) $(OBJ) $(LDFLAGS)

# Rule for object files
$(BUILD)/%.o: $(SRC)/%.c $(DEPS)
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

# Cleanup
clean:
	rm -rf $(BUILD)

# Flags
.PHONY: clean
