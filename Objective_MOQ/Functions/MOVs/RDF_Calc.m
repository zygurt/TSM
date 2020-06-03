function [ MOV ] = RDF_Calc( Model_Ref, Model_Test, Pro_Test )
%[ MOV ] = RDF_Calc( Model_Ref, Model_Test, Pro_Test )
%   As described by ITU-R BS.1387-1 Section 4.6
global debug_var

if debug_var
disp('  Relative Disturbed Frames')
end
MOV.RelDistFramesB = RelDistFramesB( Model_Ref, Model_Test, Pro_Test );


end