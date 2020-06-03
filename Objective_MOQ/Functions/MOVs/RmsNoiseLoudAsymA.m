function [ MOV] = RmsNoiseLoudAsymA(RmsNoiseLoudA, RmsMissingComponentsA)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
global debug_var

if debug_var
    disp('    RmsNoiseLoudAsymA');
end

MOV = RmsNoiseLoudA + 0.5*RmsMissingComponentsA;

end

