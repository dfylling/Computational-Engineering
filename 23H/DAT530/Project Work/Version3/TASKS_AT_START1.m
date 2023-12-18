% Consultant Company, Version 3
function [] = TASKS_AT_START()

global global_info

global_info.tasks = {'T0053'};
 
T0053.Client = 'A'; 
T0053.Size = 50;
T0053.Requirement = [1, 1, 0];
T0053.Composition = [0.5, 0.5, 0];
T0053.TimeReceived = 0;
T0053.TimeStart = '';
T0053.AssignedEmp = '';
T0053.TimeToFinish = '';
T0053.TimeFinish = '';



global_info.T0053 = T0053;
global_info.Taskpool_Size = 1; 

