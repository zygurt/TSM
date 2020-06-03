function [ MOV ] = AvgModDiff2B( Ref, Test, ModDiff2B )
%[ MOV ] = AvgModDiff2B( Ref, Test, ModDiff2B )
%   As described by ITU-R BS.1387-1 Section 4.2.2
global debug_var

if debug_var
    disp('    AvgModDiff2B');
end
Z = size(Ref.Mod,2);
w = zeros(size(Ref.Mod));
w(Test.Mod>Ref.Mod) = 1;
w(Test.Mod<Ref.Mod) = ModDiff2B.negWt;

ModDiff = w.*(abs(Test.Mod-Ref.Mod)./(ModDiff2B.offset+Ref.Mod));

ModDiff = 100*mean(ModDiff,2);

TempWt = sum(Ref.Eline./(Ref.Eline+ModDiff2B.levWt*Test.E_Thresh.^0.3),2);

%Windowed Average of Modulation Difference
[MOV,~,~] = PEAQ_Temporal_Average(ModDiff,'Linear',Z, TempWt);


end