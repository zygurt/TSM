function [ MOV ] = RmsMissingComponentsA( Ref, Test, MissingCompA, General )
%[ MOV ] = RmsMissingComponentsA( Ref, Test, NoiseLoudB, General )
%   As described by ITU-R BS.1387-1 Section 4.3.2
%Ref and Test are swapped in this compared to RmsNoiseLoudA
%The changes are made below and not in the function call
%E_Thresh is found in the Test structure
global debug_var

if debug_var
    disp('    RmsMissingComponentsA');
end
s_test = MissingCompA.ThresFac0*Ref.Mod+MissingCompA.S0;
s_ref = MissingCompA.ThresFac0*Test.Mod+MissingCompA.S0;

beta = exp(-MissingCompA.alpha.*((Ref.E_P-Test.E_P)./Test.E_P));

NL = ((Test.E_Thresh./s_test).^0.23).*...
    ((1+(max(s_test.*Ref.E_P-s_ref.*Test.E_P,0)./(Test.E_Thresh+s_ref.*Test.E_P.*beta))).^0.23-1);

%Ignore the first 50ms after reaching 0.1 sone
t = 0.1;
[~,t_r] = min(Test.N_total<t);
[~,t_t] = min(Ref.N_total<t);
ignore_frames = ceil(0.05/(General.AdvancedStepSize/General.fs))+max(t_r,t_t);

NL(1:ignore_frames,:) = 0;
for n = ignore_frames+1:size(NL,1)
    NL(n,isnan(NL(n,:))) = 0;
    NL(n,NL(n)<MissingCompA.NLmin)=0;
end

%Squared Average
[ NL_Spec ] = PEAQ_Spectral_Average( NL(ignore_frames+1:size(NL,1),:) );
[~,MOV,~] = PEAQ_Temporal_Average(24*NL_Spec,'Squared');


end

