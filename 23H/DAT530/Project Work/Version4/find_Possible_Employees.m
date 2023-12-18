% Consultant Company, Version 4
% Function returns all employees who have the certification needed to work
% on the given task
function candidates = find_Possible_Employees(task)

    global global_info
    
    % Retrieve requirements for given task
    task_requirement = global_info.(task).Requirement;
    
    % Initiate  return variable
    candidates = {};
    
    for employee = global_info.employees

        % Access employee data using the global structure
        certification = global_info.(employee{1}).Certification;

        if ~any(task_requirement - certification > 0)
            candidates{end+1} = employee{1};
        end
    end
end