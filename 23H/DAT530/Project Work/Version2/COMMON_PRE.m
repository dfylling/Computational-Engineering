% Consultant Company, Version 2
function [fire, transition] = COMMON_PRE(transition)

global global_info

% n1 = ntokens('p1'); % check how many tokens in pBuffer-1

switch transition.name
    case 't1'
         if eq(global_info.task_index, global_info.Taskpool_Size)
            granted = 0;
         else
            global_info.task_index = global_info.task_index + 1;
            taskCode = global_info.tasks(global_info.task_index);
            transition.new_color = taskCode;
            granted = 1;
         end
    case 't2'
        tokIDi = tokenAny('p2', 1);
        color = get_color('p2', tokIDi);
        transition.selected_tokens = tokIDi;

        % Access and update the global variable
        global_info.(color{1}).TimeStart = current_time();
        
        %Find and assign preferred employee
        % Hard coding 'Andy', but will make function to find preferred
        % employee in next version
        name = 'Andy';
        global_info.(color{1}).AssignedEmp = name;

        % Find task completion time and add to employee queue 
        task_size = global_info.(color{1}).Size;
        emp_speed = global_info.(name).Speed;
        task_time = round(task_size / emp_speed);
        
        % Update global variables accordingly
        global_info.(color{1}).TimeToFinish = task_time;
        global_info.(name).Queue = global_info.(name).Queue + task_time;

        % Update pritority of 'tw1', the work station
        priorset('tw1', task_time)

        % Send request for specific resource / employee
        granted = requestSR({name,1});

    case 'tw1'
        if lt(priorcomp('tw1', 't3'),1)
            granted = 0;
        else
            % Find employee assigned to current task
            tokIDi = tokenAny('p3', 1);
            color = get_color('p3', tokIDi);
            transition.selected_tokens = tokIDi;

            name = global_info.(color{1}).AssignedEmp;
    
            % Find firing time for current transition
            firing_time = get_trans('tw1').firing_time;
    
            % Subtract firing time from employee queue and priority
            global_info.(name).Queue = global_info.(name).Queue - firing_time;
 
            priordec('tw1',firing_time)
            granted = 1;
        end

    case 't3'
        % Disable transition while 'tw1', the work station has higher
        % priority
        if gt(priorcomp('tw1', 't3'),0)
            granted = 0;
        else
            tokIDi = tokenAny('p3', 1);
            color = get_color('p3', tokIDi);
            transition.selected_tokens = tokIDi;
    
            % Access and update the global variable
            global_info.(color{1}).TimeFinish = current_time();
            granted = 1;
        end
end
fire = granted;
    
    
