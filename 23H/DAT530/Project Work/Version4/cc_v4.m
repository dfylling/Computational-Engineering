%  Consultant Company, Version 4
% main simulation file (MSF) to run simulation 

clear all; clc; 
global global_info

global_info.STOP_AT = 300;

% Set the random seed for replicability
rng(123);

% Load tasks from task generator:
% N = 20;
% TASKS2(N);

% Load tasks from task list
TASKS();

% To investigate which employee is the least valuable we can simulate 
% firing either employee, {'Andy', 'Bria', 'Cass'}
global_info.fire_employee = '';

% Load employees
EMPLOYEES();

global_info.task_index = 0;

pns = pnstruct('cc_v4_pdf');

dyn.m0 = {'p6a',1, 'p6b',1, 'p6c',1};
dyn.ft = {'t1',4 , ...
          't2a',1 , 'twa',1 , 't3a',1, 't4a',1, ...
          't2b',1 , 'twb',1 , 't3b',1, 't4b',1, ...
          't2c',1 , 'twc',1 , 't3c',1, 't4c',1  ...
          };
dyn.re = {'Andy',1,inf, 'Bria',1,inf, 'Cass',1,inf}; 

pni = initialdynamics(pns, dyn); 

Sim_Results = gpensim(pni);
prnss(Sim_Results);                    
% plotp(Sim_Results, {'p2','p3a','p4'});
% plotp(Sim_Results, {'p2'}); 
plotGC(Sim_Results)

disp('Andy')
disp([{'Totaltime:', 'Totalwork:'}; num2cell([global_info.Andy.Totaltime, global_info.Andy.Totalwork])])
disp('Bria')
disp([{'Totaltime:', 'Totalwork:'}; num2cell([global_info.Bria.Totaltime, global_info.Bria.Totalwork])])
disp('Cass')
disp([{'Totaltime:', 'Totalwork:'}; num2cell([global_info.Cass.Totaltime, global_info.Cass.Totalwork])])
