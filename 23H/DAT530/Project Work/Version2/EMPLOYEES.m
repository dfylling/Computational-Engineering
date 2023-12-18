% Consultant Company, Version 2
% "index coloring" for specific resources / employees

function [] = EMPLOYEES()

global global_info

global_info.employees = {'Andy'};
 
Andy.Speed = 5; 
% global varaible .Queue will contain work time 
% queued for the certain employee and will update continuously
Andy.Queue = 0;

global_info.Andy = Andy;
global_info.Roster_Size = 1; 


