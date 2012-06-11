// ert
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <stdlib.h>
#include <stdio.h>
#include <actor/actor.h>

static actor_process_function_t main_process = ^(actor_process_t self) {
    printf("Hello World!\n");

    actor_process_sleep(self, 2.0);

    printf("Ende!\n");

    return ACTOR_SUCCESS;
};

int main(int argc, char* argv[]) {
    // error
    actor_error_t error = ACTOR_SUCCESS;

    // create node
    actor_node_t node = NULL;
    error = actor_node_create(&node, 0, 100);

    // check success
    if (error != ACTOR_SUCCESS) {
        return EXIT_FAILURE;
    }

    // start main process
    error = actor_spawn(node, NULL, main_process);

    // check success
    if (error != ACTOR_SUCCESS) {
        // cleanup
        actor_node_release(&node);

        return EXIT_FAILURE;
    }

    // wait for processes to complete
    while (actor_node_wait_for_processes(node, 10.0) != ACTOR_SUCCESS) {
        // wait
    }

    // cleanup
    actor_node_release(&node);

    return EXIT_SUCCESS;
}
