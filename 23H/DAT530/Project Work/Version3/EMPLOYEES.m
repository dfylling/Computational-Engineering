% Consultant Company, Version 3
% "index coloring" for specific resources / employees

function [] = EMPLOYEES()

global global_info

global_info.employees = {'Andy', 'Berit'};
 
Andy.Certification = [1,1,0];   % Corresponding to tasks.Requirement
Andy.Competency = [1,3,1];      % Corresponding to tasks.Composition
Andy.Queue = 0;                 % time units until all assigned work is complete

Berit.Certification = [0,1,0];
Berit.Competency = [3,3,1]; 
Berit.Queue = 0;

global_info.Andy = Andy;
global_info.Berit = Berit; 


