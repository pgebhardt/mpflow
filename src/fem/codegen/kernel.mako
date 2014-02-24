## --------------------------------------------------------------------
## This file is part of mpFlow.
##
## mpFlow is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## mpFlow is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with mpFlow. If not, see <http:#www.gnu.org/licenses/>.
##
## Copyright (C) 2014 Patrik Gebhardt
## Contact: patrik.gebhardt@rub.de
## --------------------------------------------------------------------

% if header == True:
${dtype} ${name}(
    % if custom_args is not None:
        % for i in range(len(custom_args)):
            % if i != len(custom_args) - 1:
    ${custom_args[i]},
            % else:
    ${custom_args[i]}
            % endif
        % endfor
    % else:
        % for i in range(len(args)):
            % if i != len(args) - 1:
    ${dtype} ${args[i]},
            % else:
    ${dtype} ${args[i]}
            % endif
        % endfor
    % endif
    ) {
% endif
    return ${expression};
% if header == True:
}
% endif
