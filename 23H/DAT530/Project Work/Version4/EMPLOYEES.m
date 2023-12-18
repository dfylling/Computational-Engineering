% Consultant Company, Version 4
% "index coloring" for specific resources / employees

function [] = EMPLOYEES()

global global_info

%global_info.employees = {'Andy', 'Bria', 'Cass'};
 
% EMPLOYEES_class(name, certification, competency, queue, totaltime, totalwork)
Andy = EMPLOYEES_class('Andy', [1, 1, 0], [2, 2, 1], 0, 0, 0);
Bria = EMPLOYEES_class('Bria', [0, 1, 1], [1, 1, 4], 0, 0, 0);
Cass = EMPLOYEES_class('Cass', [0, 0, 0], [4, 3, 3], 0, 0, 0);

global_info.Andy = Andy;
global_info.Bria = Bria;
global_info.Cass = Cass;

% update list of employees if any of them have been fired

if isfield(global_info, 'fire_employee') && ~isempty(global_info.fire_employee)
    switch global_info.fire_employee
        case 'Andy'
            global_info.employees = {'Bria', 'Cass'};
        case 'Bria'
            global_info.employees = {'Andy', 'Cass'};
        case 'Cass'
            global_info.employees = {'Andy', 'Bria'};
    end
else
    global_info.employees = {'Andy', 'Bria', 'Cass'};
end