% Consultant Company, Version 4
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
    case {'t2a', 't2b', 't2c'}
        % Grab a random token from 'p2'
        % Step made to avoid clogging up a workstation with a task
        % that an idle employee cannot solve
        N = ntokens('p2');
        tokIDis = tokenAny('p2', N);
        tokIDi = tokIDis(randi(N));

        color = get_color('p2', tokIDi);
        task = color{1};
        
        % Find possible employees
        possible_employees = find_Possible_Employees(task);

        % If no elible employees, do nothing
        if  isempty(possible_employees)
            granted = 0;
        else
            % Find preferred employees and task completion time
            [preferred_employee, task_time] = find_Preferred_Employee(task, possible_employees);
            
            % Request preferred employee as specific resource
            reserved = requestSR({preferred_employee,1});
    
            % If preferred employee is available update variables and fire
            if eq(reserved, 1)
                transition.selected_tokens = tokIDi;
    
                % Document time started
                global_info.(task).TimeStart = current_time();
            
                % Document employee assigned to task
                global_info.(task).AssignedEmp = preferred_employee;
            
                % Document expected time to finish task 
                global_info.(task).TimeToFinish = task_time;
                
                % Update work queue for selected employee
                global_info.(preferred_employee).Queue = global_info.(preferred_employee).Queue + task_time;
    
                % Update total time spent working by selected employee
                global_info.(preferred_employee).Totaltime = global_info.(preferred_employee).Totaltime + task_time;
    
                % Update total amount of work performed by selected employee
                task_size = global_info.(task).Size;
                global_info.(preferred_employee).Totalwork = global_info.(preferred_employee).Totalwork + task_size;
    
                % Update pritority of 'tw', the work station to simulate work
                lastChar = transition.name(end);
                tw = ['tw', lastChar];
                priorset(tw, task_time)
                granted = 1;
            else
                granted = 0;
            end
        end
    % Work station retains token as long as it has priority
    case {'twa', 'twb', 'twc'}
        lastChar = transition.name(end);
        tw = ['tw', lastChar];
        t3 = ['t3', lastChar];
        p3 = ['p3', lastChar];

        if lt(priorcomp(tw, t3),1)
            granted = 0;
        else 
            % Find employee assigned to current task
            tokIDi = tokenAny(p3, 1);
            color = get_color(p3, tokIDi);
            name = global_info.(color{1}).AssignedEmp;

            % Find firing time for current transition
            firing_time = get_trans(tw).firing_time;

            % Subtract firing time from employee queue
            global_info.(name).Queue = global_info.(name).Queue - firing_time;

            % Decrease priority of work station
            priordec(tw)
            granted = 1;
        end

    case {'t3a', 't3b', 't3c'}
        lastChar = transition.name(end);
        tw = ['tw', lastChar];
        t3 = ['t3', lastChar];
        p3 = ['p3', lastChar];

        % Disable transition while 'tw1', the work station, has higher
        % priority
        if gt(priorcomp(tw, t3),0)
            granted = 0;
        else
            tokIDi = tokenAny(p3, 1);
            color = get_color(p3, tokIDi);
            transition.selected_tokens = tokIDi;
    
            % Document task finishing time
            global_info.(color{1}).TimeFinish = current_time();
            granted = 1;
        end
    case {'t4a', 't4b', 't4c'}
        transition.override = 1;
        granted = 1;
end
fire = granted;
    
    
