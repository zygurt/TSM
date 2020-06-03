function [ MOV ] = NL_Calc( Pro_Ref, Pro_Test, Ref_Model_FB, FB_Ref_Prepro, FB_Test_Prepro, General )
%[ MOV ] = NL_Calc( Pro_Ref, Pro_Test, General )
%   As described by ITU-R BS.1387-1 Section 4.3
%   Only 'Basic' Feature is implemented
global debug_var

if debug_var
disp('  Noise Loudness')
end
MissingCompB.alpha = 1.5;
MissingCompB.ThresFac0 = 0.15;
MissingCompB.S0 = 1;
MissingCompB.NLmin = 0;

NoiseLoudB.alpha = 1.5;
NoiseLoudB.ThresFac0 = 0.15;
NoiseLoudB.S0 = 0.5;
NoiseLoudB.NLmin = 0;

MissingCompA.alpha = 1.5;
MissingCompA.ThresFac0 = 0.15;
MissingCompA.S0 = 1;
MissingCompA.NLmin = 0;

NoiseLoudA.alpha = 2.5;
NoiseLoudA.ThresFac0 = 0.3;
NoiseLoudA.S0 = 1;
NoiseLoudA.NLmin = 0.1;

LinDist.alpha = 1.5;
LinDist.ThresFac0 = 0.15;
LinDist.S0 = 1;
LinDist.NLmin = 0;


[ MOV.RmsNoiseLoudB ] = RmsNoiseLoudB( Pro_Ref, Pro_Test, NoiseLoudB, General );
[ MOV.RmsNoiseLoudA ] = RmsNoiseLoudA(  FB_Ref_Prepro, FB_Test_Prepro, NoiseLoudA, General );
[ MOV.RmsMissingComponentsA ] = RmsMissingComponentsA( FB_Ref_Prepro, FB_Test_Prepro, MissingCompA, General );
[ MOV.RmsNoiseLoudAsymA] = RmsNoiseLoudAsymA(MOV.RmsNoiseLoudA, MOV.RmsMissingComponentsA);

[ MOV.AvgLinDistA ] = AvgLinDistA(Ref_Model_FB, FB_Ref_Prepro, FB_Test_Prepro, LinDist, General);


end

