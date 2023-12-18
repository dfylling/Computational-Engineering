% Consultant Company, Version 4
% Employee class

classdef TASKS_class
    properties
        Client
        Size
        Requirement
        Composition
        TimeReceived
        TimeStart
        AssignedEmp
        TimeToFinish
        TimeFinish
    end
    
    methods
        function obj = TASKS_class(client, size, requirement, composition)
            obj.Client = client;
            obj.Size = size;
            obj.Requirement = requirement;
            obj.Composition = composition;
            obj.TimeReceived = '';
            obj.TimeStart = '';
            obj.AssignedEmp = '';
            obj.TimeToFinish = '';
            obj.TimeFinish = '';
        end
    end
end