function [ BW ] = BW_Calc( Model_Ref, Model_Test, General )
%UNTITLED Summary of this function goes here
%   As described by ITU-R BS.1387-1 Section 4.4
global debug_var

if debug_var
disp('  Bandwidth')
end
BW = PEAQ_Bandwidth(Model_Ref.X_MAG, Model_Test.X_MAG, General);

end

