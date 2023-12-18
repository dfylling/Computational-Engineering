%  Consultant Company, Version 1
% main simulation file (MSF) to run simulation 

clear all; clc; 
global global_info
global_info.STOP_AT = 10;

% load the initial Tasks
TASKS_AT_START();

global_info.task_index = 0;

pns = pnstruct('cc_v1_pdf');

dyn.m0 = {};
dyn.ft = {'t1',5 , 't2',1}; 

pni = initialdynamics(pns, dyn); 

Sim_Results = gpensim(pni);             % perform simulation runs
prnss(Sim_Results);                     % print the simulation results 
plotp(Sim_Results, {'p2','p3'});        % plot the results

disp(global_info.T0053)
