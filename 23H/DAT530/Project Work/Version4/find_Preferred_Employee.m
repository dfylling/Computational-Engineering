% Consultant Company, Version 4
% Function returns the employee who will finish the task first based on the
% composition of the task and the competency of the employee

function [preferred_employee, task_time_return] = find_Preferred_Employee(task, possible_employees)

    global global_info

    % Retrieve task size from global memory
    task_size = global_info.(task).Size;
    task_composition = global_info.(task).Composition;
    
    record_time = 10000;
    % Initialize return variable
    preferred_employee = [];

    % Iterate through possible employees
    for employee = possible_employees

        % Work speed is calculated from the relaationship between employee
        % competency and composition of given task
        speed = sum(times(global_info.(employee{1}).Competency, task_composition));
        queue = global_info.(employee{1}).Queue;

        % Calculate time taken for employee to finish current queue and new
        % task
        task_time = task_size / speed;
        total_time = queue + task_time;

        %Fastest employee is chosen
        if total_time < record_time
            preferred_employee = employee{1};
            task_time_return = round(task_time);
            record_time = total_time;
        end
    end
end


