% Consultant Company, Version 4
function [] = TASKS()

global global_info

global_info.tasks = {'T001', 'T002', 'T003', 'T004'};

T001 = TASKS_class('A', 10, [0, 1, 0], [0.5, 0.5, 0.0]);
T002 = TASKS_class('A', 30, [1, 0, 0], [0.2, 0.6, 0.2]);
T003 = TASKS_class('B', 60, [0, 0, 0], [0.5, 0.5, 0.0]);
T004 = TASKS_class('B', 90, [0, 0, 1], [0.1, 0.0, 0.9]);

global_info.T001 = T001;
global_info.T002 = T002;
global_info.T003 = T003;
global_info.T004 = T004;
global_info.Taskpool_Size = 4; 

