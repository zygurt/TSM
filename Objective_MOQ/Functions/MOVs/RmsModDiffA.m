function [ MOV ] = RmsModDiffA( Ref, Test, ModDiffA )
%[ MOV ] = AvgModDiff2B( Ref, Test, ModDiffA )
%   As described by ITU-R BS.1387-1 Section 4.2.2
global debug_var

if debug_var
    disp('    RmsModDiffA');
end
Z = size(Ref.Mod,2);
w = ones(size(Ref.Mod));
% The next three lines are replaced by the 1 above
% w = zeros(size(Ref.Mod));
% w(Test.Mod>Ref.Mod) = 1;
% w(Test.Mod<Ref.Mod) = ModDiffA.negWt;

ModDiff = w.*(abs(Test.Mod-Ref.Mod)./(ModDiffA.offset+Ref.Mod));

ModDiff = 100*mean(ModDiff,2);

TempWt = sum(Ref.Eline./(Ref.Eline+ModDiffA.levWt*Test.E_Thresh.^0.3),2);

%Windowed Average of Modulation Difference
[~,MOV,~] = PEAQ_Temporal_Average(ModDiff,'Squared',Z, TempWt);


end