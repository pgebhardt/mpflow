# libactor
#
# Implementation of an erlang style actor model using libdispatch
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
LDFLAGS = 

# Directories
SRC = src
BUILD = build

# Install directories
INCLUDES = /usr/local/include/actor
LIBS = /usr/local/lib

# Object files
_OBJ = actor.o message.o process.o node.o distributer.o error.o
OBJ = $(patsubst %, $(BUILD)/%, $(_OBJ))

# Dependencies
_DEPS = actor.h message.h process.h node.h distributer.h error.h common.h
DEPS = $(patsubst %, $(SRC)/%, $(_DEPS))

# Output file
BIN = libactor.a

# Rule for library
$(BIN): $(OBJ) $(DEPS)
	mkdir -p $(BUILD)
	ar rc $(BUILD)/$(BIN) $(OBJ)
	ranlib $(BUILD)/$(BIN)

# Rule for object files
$(BUILD)/%.o: $(SRC)/%.c $(DEPS)
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

# Cleanup
clean:
	rm -rf $(BUILD)

# Install
install: $(BIN)
	mkdir -p $(INCLUDES)
	install -m 0644 $(SRC)/*.h $(INCLUDES)
	install -m 0644 $(BUILD)/$(BIN) $(LIBS)

# Uninstall
uninstall:
	rm -rf $(INCLUDES)
	rm -rf $(LIBS)/$(BIN)

# Flags
.PHONY: clean install
