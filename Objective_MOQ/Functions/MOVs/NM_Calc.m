function [ MOV ] = NM_Calc( Model_Ref, Model_Test, Pro_Test )
%[ MOV ] = NM_Calc( Model_Ref, Model_Test, Pro_Test )
%   As described by ITU-R BS.1387-1 Section 4.5
global debug_var

if debug_var
    disp('  Noise to Mask Ratio')
end
MOV.TotalNMRB = TotalNMRB( Model_Ref, Model_Test, Pro_Test );

MOV.SegmentalNMRB = SegmentalNMRB(Model_Ref, Model_Test, Pro_Test);


end

