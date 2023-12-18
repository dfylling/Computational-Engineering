% Consultant Company, Version 3
function [fire, transition] = COMMON_POST(transition)

global global_info

switch transition.name
    case {'t1', 't2'}
        % Do nothing.
    case 'tw1'
        % % Find employee assigned to current task
        % tokIDi = tokenAny('p3', 1);
        % color = get_color('p3', tokIDi);
        % name = global_info.(color{1}).AssignedEmp;
        % 
        % % Find firing time for current transition
        % firing_time = get_trans('tw1').firing_time;
        % 
        % % Subtract firing time from employee queue
        % global_info.(name).Queue = global_info.(name).Queue - firing_time;
    case 't3'
        release('t2')
end
