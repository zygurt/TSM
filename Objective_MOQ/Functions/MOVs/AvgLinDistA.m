function [ MOV ] = AvgLinDistA(Ref_Model_FB, Ref, Test, LinDist, General)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
global debug_var

if debug_var
    disp('    AvgLinDistA');
end

s_test = LinDist.ThresFac0*Ref.Mod+LinDist.S0; %ITU is not specific.  About what to use for Mod here
%Given that the non-spectrally excited patterns are being used, an
%appropriate Mod doesn't exist.  Using the Reference Mod.
s_ref = LinDist.ThresFac0*Ref.Mod+LinDist.S0;

beta = exp(-LinDist.alpha.*((Ref_Model_FB.E-Ref.E_P)./Ref.E_P));

NL = ((Test.E_Thresh./s_test).^0.23).*...
    ((1+(max(s_test.*Ref_Model_FB.E-s_ref.*Ref.E_P,0)./(Test.E_Thresh+s_ref.*Ref.E_P.*beta))).^0.23-1);

%Ignore the first 50ms after reaching 0.1 sone
t = 0.1;
[~,t_r] = min(Ref.N_total<t);
[~,t_t] = min(Test.N_total<t);
ignore_frames = ceil(0.05/(General.AdvancedStepSize/General.fs))+max(t_r,t_t);

NL(1:ignore_frames,:) = 0;
for n = ignore_frames+1:size(NL,1)
    NL(n,isnan(NL(n,:))) = 0;
    NL(n,NL(n)<LinDist.NLmin)=0;
end

%Squared Average
[ NL_Spec ] = PEAQ_Spectral_Average( NL(ignore_frames+1:size(NL,1),:) );
[~,MOV,~] = PEAQ_Temporal_Average(24*NL_Spec,'Squared');



end

