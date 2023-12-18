% Consultant Company, Version 2
function [fire, transition] = COMMON_POST(transition)

global global_info

switch transition.name
    case {'t1', 't2', 'tw1'}
        % Do nothing.
    case 't3'
        release('t2')
end
