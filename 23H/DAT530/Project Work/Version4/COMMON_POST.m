% Consultant Company, Version 4
function [fire, transition] = COMMON_POST(transition)

switch transition.name
    case {'t1'}
        % Do nothing.
    case {'t3a', 't3b', 't3c'}
        lastChar = transition.name(end);
        t2 = ['t2', lastChar];
        release(t2)
end
