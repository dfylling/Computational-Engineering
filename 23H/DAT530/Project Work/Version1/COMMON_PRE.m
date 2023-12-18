% Consultant Company, Version 1
function [fire, transition] = COMMON_PRE(transition)

global global_info

% n1 = ntokens('p1'); % check how many tokens in pBuffer-1

switch transition.name
    case 't1'
         if eq(global_info.task_index, global_info.Taskpool_Size)
            fire = 0;
         else
            global_info.task_index = global_info.task_index + 1;
            taskCode = global_info.tasks(global_info.task_index);
            transition.new_color = taskCode;
            fire = 1;
         end

    case 't2'
        tokIDi = tokenAny('p2', 1);
        color = get_color('p2', tokIDi);

        % Access and update the global variable
        global_info.(color{1}).TimeStart = current_time();
        fire = 1;
        end
end

    
    
