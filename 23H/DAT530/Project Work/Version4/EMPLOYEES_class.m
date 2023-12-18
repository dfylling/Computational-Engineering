% Consultant Company, Version 4
% Employee class

classdef EMPLOYEES_class
    properties
        Name
        Certification
        Competency
        Queue
        Totaltime
        Totalwork
    end
    methods
        function obj = EMPLOYEES_class(name, certification, competency, queue, totaltime, totalwork)
            obj.Name = name;
            obj.Certification = certification;
            obj.Competency = competency;
            obj.Queue = queue;
            obj.Totaltime = totaltime;
            obj.Totalwork = totalwork;
        end
    end
end


