function [ MOV ] = ModDiff( Pro_Ref, Pro_Test)
%[ MOV ] = ModDiff( Pro_Ref, Pro_Test)
%   As described by ITU-R BS.1387-1 Section 4.2

ModDiff1B.negWt = 1;
ModDiff1B.offset = 1;
ModDiff1B.levWt = 100;
ModDiff2B.negWt = 0.1;
ModDiff2B.offset = 0.01;
ModDiff2B.levWt = 100;
ModDiffA.negWt = 1;
ModDiffA.offset = 1;
ModDiffA.levWt = 1;


[ MOV.WinModDiff1B ] = WinModDiff1B( Pro_Ref.Mod, Pro_Test.Mod, ModDiff1B );
[ MOV.AvgModDiff1B ] = AvgModDiff1B( Pro_Ref, Pro_Test, ModDiff1B );
[ MOV.AvgModDiff2B ] = AvgModDiff2B( Pro_Ref, Pro_Test, ModDiff2B );

end

