function [ MOV ] = RmsNoiseLoudB( Ref, Test, NoiseLoudB, General )
%[ MOV ] = RmsNoiseLoudB( Ref, Test, NoiseLoudB, General )
%   As described by ITU-R BS.1387-1 Section 4.3.5
global debug_var

if debug_var
    disp('    RMSNoiseLoudB');
end
s_test = NoiseLoudB.ThresFac0*Test.Mod+NoiseLoudB.S0;
s_ref = NoiseLoudB.ThresFac0*Ref.Mod+NoiseLoudB.S0;

beta = exp(-NoiseLoudB.alpha.*((Test.E_P-Ref.E_P)./Ref.E_P));

NL = ((Test.E_Thresh./s_test).^0.23).*...
    ((1+(max(s_test.*Test.E_P-s_ref.*Ref.E_P,0)./(Test.E_Thresh+s_ref.*Ref.E_P.*beta))).^0.23-1);

%Ignore the first 50ms after reaching 0.1 sone
t = 0.1;
[~,t_r] = min(Ref.N_total<t);
[~,t_t] = min(Test.N_total<t);
ignore_frames = ceil(0.05/(General.BasicStepSize/General.fs))+max(t_r,t_t);

NL(1:ignore_frames,:) = 0;
for n = ignore_frames+1:size(NL,1)
    NL(n,isnan(NL(n,:))) = 0;
    NL(n,NL(n)<NoiseLoudB.NLmin)=0;
end

%Squared Average
[ NL_Spec ] = PEAQ_Spectral_Average( NL(ignore_frames+1:size(NL,1),:) );
[~,MOV,~] = PEAQ_Temporal_Average(24*NL_Spec,'Squared');


end

