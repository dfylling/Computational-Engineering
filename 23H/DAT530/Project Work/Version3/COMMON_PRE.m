% Consultant Company, Version 3
function [fire, transition] = COMMON_PRE(transition)

global global_info

switch transition.name
    case 't1'
         if eq(global_info.task_index, global_info.Taskpool_Size)
            granted = 0;
         else
            global_info.task_index = global_info.task_index + 1;
            color = global_info.tasks(global_info.task_index);
            transition.new_color = color;
            
            % Document time when task is put into 'p2'
            firing_time = get_trans('t1').firing_time;
            global_info.(color{1}).TimeReceived = current_time() + firing_time;
            granted = 1;
         end
    case 't2'
        tokIDi = tokenAny('p2', 1);
        color = get_color('p2', tokIDi);
        task = color{1};
        
        % Find possible employees
        possible_employees = find_Possible_Employees(task);

        % Find preferred employees and task completion time
        [preferred_employee, task_time] = find_Preferred_Employee(task, possible_employees);
        
        % Request preferred employee as specific resource
        reserved = requestSR({preferred_employee,1});

        % If preferred employee is available reserved = 1 and update variables
        if eq(reserved, 1)
            transition.selected_tokens = tokIDi;

            % Document time started
            global_info.(color{1}).TimeStart = current_time();
        
            % Document employee assigned to task
            global_info.(color{1}).AssignedEmp = preferred_employee;
        
            % Document expected time to finish task 
            global_info.(color{1}).TimeToFinish = task_time;
            
            % Update work queue for selected employee
            global_info.(preferred_employee).Queue = global_info.(preferred_employee).Queue + task_time;

            % Update pritority of 'tw1', the work station to simulate work
            priorset('tw1', task_time)
            granted = 1;
        else
            granted = 0;
        end

    % Work station retains token as long as it has priority
    case 'tw1'
        if lt(priorcomp('tw1', 't3'),1)
            granted = 0;
        else 
            % Find employee assigned to current task
            tokIDi = tokenAny('p3', 1);
            color = get_color('p3', tokIDi);
            name = global_info.(color{1}).AssignedEmp;

            % Find firing time for current transition
            firing_time = get_trans('tw1').firing_time;

            % Subtract firing time from employee queue
            global_info.(name).Queue = global_info.(name).Queue - firing_time;
            % Decrease priority of work station
            priordec('tw1')
            granted = 1;
        end

    case 't3'
        % Disable transition while 'tw1', the work station, has higher
        % priority
        if gt(priorcomp('tw1', 't3'),0)
            granted = 0;
        else
            tokIDi = tokenAny('p3', 1);
            color = get_color('p3', tokIDi);
            transition.selected_tokens = tokIDi;
    
            % Document task finishing time
            global_info.(color{1}).TimeFinish = current_time();
            granted = 1;
        end
    case 't4'
        transition.override = 1;
        granted = 1;
end
fire = granted;
    
    
