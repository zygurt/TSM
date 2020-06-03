function [ MOV ] = EHS_Calc( Pro_Test, General )
%[ MOV ] = EHS_Calc( Pro_Test, General )
%   As described by ITU-R BS.1387-1 Section 4.8
global debug_var

if debug_var
disp('  Harmonic Structure of Error');
end
MOV.EHSB = EHSB(Pro_Test, General);

end

