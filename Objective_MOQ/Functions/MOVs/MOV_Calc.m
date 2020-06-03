function [ MOV ] = MOV_Calc( Model_Ref, Model_Test, Pro_Ref, Pro_Test, Ref_Model_FB, Test_Model_FB, FB_Ref_Prepro, FB_Test_Prepro, General )
%[ MOV ] = MOV_Calc( Model_Ref, Model_Test, Pro_Ref, Pro_Test, General )
%   Calculation of the Model Output Variables
%   As described by ITU-R BS.1387-1 Section 4
global debug_var

if debug_var
disp('Model Output Variable Calculation')
end
%Basic Features
[ Diff ] = ModDiff_Calc( Pro_Ref, Pro_Test, FB_Ref_Prepro, FB_Test_Prepro);
[ NL ] = NL_Calc( Pro_Ref, Pro_Test, Ref_Model_FB, FB_Ref_Prepro, FB_Test_Prepro, General );
[ BW ] = BW_Calc( Model_Ref, Model_Test, General );
[ NM ] = NM_Calc( Model_Ref, Model_Test, Pro_Test );
[ RDF ] = RDF_Calc( Model_Ref, Model_Test, Pro_Test );
[ DP ] = DP_Calc(Model_Ref, Model_Test, General);
[ EHS ] = EHS_Calc(Pro_Test, General);

MOV.WinModDiff1B = Diff.WinModDiff1B;
MOV.AvgModDiff1B = Diff.AvgModDiff1B;
MOV.AvgModDiff2B = Diff.AvgModDiff2B;
MOV.RmsNoiseLoudB = NL.RmsNoiseLoudB;
MOV.BandwidthRefB = BW.BandwidthRefB;
MOV.BandwidthTestB = BW.BandwidthTestB;
MOV.BandwidthTestB_new = BW.BandwidthTestB_new;
MOV.TotalNMRB = NM.TotalNMRB;

MOV.RelDistFramesB = RDF.RelDistFramesB;
MOV.MFPDB = DP.MFPDB;
MOV.ADBB = DP.ADBB;
MOV.EHSB = EHS.EHSB;

%Advanced Features
MOV.RmsModDiffA = Diff.RmsModDiffA;
MOV.RmsNoiseLoudAsymA = NL.RmsNoiseLoudAsymA;
MOV.AvgLinDistA = NL.AvgLinDistA;
MOV.SegmentalNMRB = NM.SegmentalNMRB; %Only used in Advanced version.


end

