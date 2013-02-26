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
% for expression in reversed(expressions):
    ${dtype} ${expression[0]} = ${expression[1]};
% endfor
    return ${expressions[0][0]};
% if header == True:
}
% endif
