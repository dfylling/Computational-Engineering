%  Consultant Company, Version 3
% main simulation file (MSF) to run simulation 

clear all; clc; 
global global_info
global_info.STOP_AT = 100;

% load the initial Tasks
TASKS_AT_START();
% and employees
EMPLOYEES();

global_info.task_index = 0;

pns = pnstruct('cc_v3_pdf');

dyn.m0 = {'p6',1};
dyn.ft = {'t1',5 , 't2',1 , 'tw1',1 , 't3',1, 't4',1};
dyn.re = {'Andy',1,inf, 'Berit',1,inf}; 

pni = initialdynamics(pns, dyn); 

Sim_Results = gpensim(pni);             % perform simulation runs
prnss(Sim_Results);                     % print the simulation results 
plotp(Sim_Results, {'p2','p3','p4'});
plotp(Sim_Results, {'p2'}); % plot the results
plotGC(Sim_Results)

disp(global_info.T0053)
disp(global_info.T0066)
disp(global_info.Andy)
disp(global_info.Berit)
