% Consultant Company, Version 4
function [] = TASKS2(N)
% Generates N random tasks based on input given below

global global_info

% Initialize the global_info structure
global_info.Taskpool_Size = N;
global_info.tasks = cell(1, N);

% Populate the global_info structure
for taskNum = 1:N
    % Generate random properties
    clientOptions = ['A', 'B', 'C'];
    client = clientOptions(randi(length(clientOptions)));
    size = randi([10, 30]);
    requirement = [0, 0, 0];
    if rand() < 0.6
        requirement(randi(3)) = 1;
    end
    composition = [rand, rand, rand];
    composition = composition / sum(composition);

    % Create the task and assign it to the global_info structure
    taskName = ['T', num2str(taskNum, '%03d')];
    task = TASKS_class(client, size, requirement, composition);
    global_info.(taskName) = task;
    global_info.tasks{taskNum} = taskName;
end

