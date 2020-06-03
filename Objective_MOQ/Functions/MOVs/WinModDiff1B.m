function [ MOV ] = WinModDiff1B( Mod_ref, Mod_test, ModDiff1B )
%UNTITLED2 Summary of this function goes here
%   As described by ITU-R BS.1387-1 Section 4.2.2
global debug_var

if debug_var
    disp('    WinModDiff1B');
end
Z = size(Mod_ref,2);
w = zeros(size(Mod_ref));
w(Mod_test>Mod_ref) = 1;
w(Mod_test<Mod_ref) = ModDiff1B.negWt;

ModDiff = w.*(abs(Mod_test-Mod_ref)./(ModDiff1B.offset+Mod_ref));

ModDiff = 100*mean(ModDiff,2);

%Windowed Average of Modulation Difference
[~,~,MOV] = PEAQ_Temporal_Average(ModDiff,'Windowed',Z);


end

