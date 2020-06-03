function [ OMOV, OMOV_name ] = OMOQ( ref, test, side_data, match_method )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
global debug_var

TSM = side_data.TSM;
MeanOS = side_data.MeanOS;
MedianOS = side_data.MedianOS;
% StdOS = side_data.StdOS;
MeanOS_RAW = side_data.MeanOS_RAW;
MedianOS_RAW = side_data.MedianOS_RAW;
% StdOS_RAW = side_data.StdOS_RAW;

[sig_ref, ~] = audioread(ref);
[sig_test, General.fs] = audioread(test);

General.version = 'basic';
General.model = 'FFT';
General.N = 2048;
General.BasicStepSize = 1024;
General.AdvancedStepSize = 192;
General.TSM = TSM;
General.Testname = test;

%PEAQ Implementation
%Basic has been implemented, Advanced is partially implemented.
%Prepare the signal
[sig_ref, sig_test, General] = Signal_Prep(sig_ref, sig_test, General);

% %Additional TSM Features
[Ph_NW, Ph_MW] = Phasiness4(sig_ref, sig_test, General);
[SS_MAD, SS_MD] = Spec_Sim(sig_ref, sig_test);
[peak_delta, transient_ratio, hpsep_transient_ratio] = Transientness(sig_ref, sig_test, General);

% MusNoise_framing = Spectral_Kurtosis(sig_ref, sig_test, General, 'framing');
% MusNoise_up = Spectral_Kurtosis(sig_ref, sig_test, General, 'up');
% MusNoise_down = Spectral_Kurtosis(sig_ref, sig_test, General, 'down');
% MusNoise_to_ref = Spectral_Kurtosis(sig_ref, sig_test, General, 'to_ref');
% MusNoise_to_test = Spectral_Kurtosis(sig_ref, sig_test, General, 'to_test');


% Ear Model
if (exist('Ear.mat', 'file') == 2 && debug_var)
    disp('Using Previously Generated Ear Model file');
    load('Ear.mat')
else
    [ Ref_Model_FB, Test_Model_FB ] = fb_Ear_Model(sig_ref, sig_test, General, match_method);
    [ Ref_Model ] = fft_Ear_Model( sig_ref, General, match_method, 'ref');
    [ Test_Model ] = fft_Ear_Model( sig_test, General, match_method, 'test');
%     fprintf('Ref frames = %d, Test frames = %d\n',size(Ref_Model.X_MAG,2),size(Test_Model.X_MAG,2));
    if debug_var
        save('Ear.mat','Ref_Model','Test_Model','Ref_Model_FB','Test_Model_FB','General')%, 'Ref_Model_FB');
    end
end

%Pre-Processing of excitation patterns
if (exist('Pre-pro.mat', 'file') == 2 && debug_var)
    disp('Using Previously Generated Pre-Processing file');
    load('Pre-pro.mat')
else
    [FB_Ref_Prepro, FB_Test_Prepro] = fb_pre_pro( Ref_Model_FB, Test_Model_FB, General );
    [Ref_Prepro, Test_Prepro] = fft_pre_pro( Ref_Model, Test_Model, General);
    
    if debug_var
        save('Pre-pro.mat', 'Ref_Prepro', 'Test_Prepro', 'FB_Ref_Prepro', 'FB_Test_Prepro');
    end
end


%Calculation of Model Output Variables
MOV = MOV_Calc( Ref_Model, Test_Model, Ref_Prepro, Test_Prepro, Ref_Model_FB, Test_Model_FB, FB_Ref_Prepro, FB_Test_Prepro, General );

% MOV.TSM = TSM;
% MOV.MeanOS = MeanOS;
% MOV.MedianOS = MedianOS;
% MOV.StdOS = StdOS;

%Additional Measures of Quality
%Previous Objective Measures
MOV.DM = DM_Calc( Ref_Model, Test_Model);
MOV.SER = SER_Calc( Ref_Model, Test_Model);

%Transient Features
MOV.peak_delta = peak_delta;
MOV.transient_ratio = transient_ratio;
MOV.hpsep_transient_ratio = hpsep_transient_ratio;

%Musical Noise Features
% MOV.MusNoise_framing_l = MusNoise_framing(1);
% MOV.MusNoise_framing_m = MusNoise_framing(2);
% MOV.MusNoise_framing_u = MusNoise_framing(3);
% MOV.MusNoise_framing_max = MusNoise_framing(4);
% MOV.MusNoise_up_l = MusNoise_up(1);
% MOV.MusNoise_up_m = MusNoise_up(2);
% MOV.MusNoise_up_u = MusNoise_up(3);
% MOV.MusNoise_up_max = MusNoise_up(4);
% MOV.MusNoise_down_l = MusNoise_down(1);
% MOV.MusNoise_down_m = MusNoise_down(2);
% MOV.MusNoise_down_u = MusNoise_down(3);
% MOV.MusNoise_down_max = MusNoise_down(4);
% MOV.MusNoise_to_ref_l = MusNoise_to_ref(1);
% MOV.MusNoise_to_ref_m = MusNoise_to_ref(2);
% MOV.MusNoise_to_ref_u = MusNoise_to_ref(3);
% MOV.MusNoise_to_ref_max = MusNoise_to_ref(4);
% MOV.MusNoise_to_test_l = MusNoise_to_test(1);
% MOV.MusNoise_to_test_m = MusNoise_to_test(2);
% MOV.MusNoise_to_test_u = MusNoise_to_test(3);
% MOV.MusNoise_to_test_max = MusNoise_to_test(4);


%Phasiness Features

MOV.ave_P_ave_f = Ph_NW.ave_P_ave_f;
MOV.std_P_ave_f = Ph_NW.std_P_ave_f; 

% % MOV.ave_P_ave_t = Ph.ave_P_ave_t;
% % MOV.Ph_m_t =  Ph.Ph_m_t;
% % MOV.P_ave_end = Ph.P_ave_end;
% % MOV.P_std_end = Ph.P_std_end;
% % MOV.ave_Ph_mad_t_diff = Ph.ave_Ph_mad_t_diff;


MOV.ave_P_ave_f_mag = Ph_MW.ave_P_ave_f_mag;
MOV.std_P_ave_f_mag = Ph_MW.std_P_ave_f_mag; 

% % MOV.ave_P_ave_t_mag = Ph_mag.ave_P_ave_t_mag;
% % MOV.Ph_m_t_mag =  Ph_mag.Ph_m_t_mag;
% % MOV.P_ave_end_mag = Ph_mag.P_ave_end_mag;
% % MOV.P_std_end_mag = Ph_mag.P_std_end_mag;
% % MOV.ave_Ph_mad_t_diff_mag = Ph_mag.ave_Ph_mad_t_diff_mag;

% % MOV.ave_P_ave_t_pow = Ph_pow.ave_P_ave_t_pow;
% % MOV.ave_P_ave_f_pow = Ph_pow.ave_P_ave_f_pow;
% % MOV.std_P_ave_f_pow = Ph_pow.std_P_ave_f_pow; 
% % MOV.Ph_m_t_pow =  Ph_pow.Ph_m_t_pow;
% % MOV.P_ave_end_pow = Ph_pow.P_ave_end_pow;
% % MOV.P_std_end_pow = Ph_pow.P_std_end_pow;
% % MOV.ave_Ph_mad_t_diff_pow = Ph_pow.ave_Ph_mad_t_diff_pow;

% % MOV.ave_P_ave_t_fm = Ph_fm.ave_P_ave_t_fm;
% % MOV.ave_P_ave_f_fm = Ph_fm.ave_P_ave_f_fm;
% % MOV.std_P_ave_f_fm = Ph_fm.std_P_ave_f_fm; 
% % MOV.Ph_m_t_fm =  Ph_fm.Ph_m_t_fm;
% % MOV.P_ave_end_fm = Ph_fm.P_ave_end_fm;
% % MOV.P_std_end_fm = Ph_fm.P_std_end_fm;
% % MOV.ave_Ph_mad_t_diff_fm = Ph_fm.ave_Ph_mad_t_diff_fm;

% Spectral Similarity Feature
MOV.SS_MAD = SS_MAD;
MOV.SS_MD = SS_MD;

% %Used when testing musical noise features
% OMOV = [MeanOS, MedianOS, StdOS, ...
%         MeanOS_RAW, MedianOS_RAW, StdOS_RAW, ...
%         TSM, ...
%         MOV.peak_delta, MOV.transient_ratio, MOV.hpsep_transient_ratio];
% 
% 
% OMOV_name = {'MeanOS', 'MedianOS', 'StdOS', ...
%              'MeanOS_RAW', 'MedianOS_RAW', 'StdOS_RAW', ...
%              'TSM', ...
%              'peak_delta', 'transient_ratio', 'hpsep_transient_ratio'};

% OMOV = [MOV.MeanOS, MOV.MedianOS, MOV.StdOS, TSM, ...
%     MOV.WinModDiff1B, MOV.AvgModDiff1B, MOV.AvgModDiff2B, ...
%     MOV.RmsNoiseLoudB, ...
%     MOV.BandwidthRefB, MOV.BandwidthTestB, MOV.BandwidthTestB_new, ...
%     MOV.TotalNMRB, ...
%     MOV.RelDistFramesB, ...
%     MOV.MFPDB, MOV.ADBB, ...
%     MOV.EHSB, ...
%     MOV.DM, MOV.SER, ...
%     MOV.peak_delta, MOV.transient_ratio, MOV.hpsep_transient_ratio, ...
%     MOV.ave_P_ave_t, MOV.ave_P_ave_f, MOV.std_P_ave_f,  MOV.Ph_m_t, MOV.P_ave_end, MOV.P_std_end, MOV.ave_Ph_mad_t_diff, ...
%     MOV.ave_P_ave_t_mag, MOV.ave_P_ave_f_mag, MOV.std_P_ave_f_mag,  MOV.Ph_m_t_mag, MOV.P_ave_end_mag, MOV.P_std_end_mag, MOV.ave_Ph_mad_t_diff_mag, ...
%     MOV.ave_P_ave_t_pow, MOV.ave_P_ave_f_pow, MOV.std_P_ave_f_pow,  MOV.Ph_m_t_pow, MOV.P_ave_end_pow, MOV.P_std_end_pow, MOV.ave_Ph_mad_t_diff_pow, ...
%     MOV.ave_P_ave_t_fm, MOV.ave_P_ave_f_fm, MOV.std_P_ave_f_fm,  MOV.Ph_m_t_fm, MOV.P_ave_end_fm, MOV.P_std_end_fm, MOV.ave_Ph_mad_t_diff_fm, ...
%     MOV.MusNoise_framing_l, MOV.MusNoise_framing_m, MOV.MusNoise_framing_u, MOV.MusNoise_framing_max, ...
%     MOV.MusNoise_up_l, MOV.MusNoise_up_m, MOV.MusNoise_up_u, MOV.MusNoise_up_max, ...
%     MOV.MusNoise_down_l, MOV.MusNoise_down_m, MOV.MusNoise_down_u, MOV.MusNoise_down_max, ...
%     MOV.MusNoise_to_ref_l, MOV.MusNoise_to_ref_m, MOV.MusNoise_to_ref_u, MOV.MusNoise_to_ref_max, ...
%     MOV.MusNoise_to_test_l, MOV.MusNoise_to_test_m, MOV.MusNoise_to_test_u, MOV.MusNoise_to_test_max];
% 
% 
% OMOV_name = {'MeanOS', 'MedianOS', 'StdOS', 'TSM', ...
%     'WinModDiff1B', 'AvgModDiff1B', 'AvgModDiff2B', ...
%     'RmsNoiseLoudB', ...
%     'BandwidthRefB', 'BandwidthTestB', 'BandwidthTestB_new', ...
%     'TotalNMRB', ...
%     'RelDistFramesB', ...
%     'MFPDB', 'ADBB', ...
%     'EHSB', ...
%     'DM', 'SER', ...
%     'peak_delta', 'transient_ratio', 'hpsep_transient_ratio', ...
%     'M-MTAPDNW', 'M-MFAPDNW', 'S-MFAPDNW',  'M-MFPDNW', 'MFAPD-ENDNW', 'SFAPD-ENDNW', 'M-FDMTAPDNW', ...
%     'M-MTAPDMW', 'M-MFAPDMW', 'S-MFAPDMW',  'M-MFPDMW', 'MFAPD-ENDMW', 'SFAPD-ENDMW', 'M-FDMTAPDMW', ...
%     'M-MTAPDPW', 'M-MFAPDPW', 'S-MFAPDPW',  'M-MFPDPW', 'MFAPD-ENDPW', 'SFAPD-ENDPW', 'M-FDMTAPDPW', ...
%     'M-MTAPDFMW', 'M-MFAPDFMW', 'S-MFAPDFMW',  'M-MFPDFMW', 'MFAPD-ENDFMW', 'SFAPD-ENDFMW', 'M-FDMTAPDFMW', ...
%     'MusNoise_framing_l', 'MusNoise_framing_m', 'MusNoise_framing_u', 'MusNoise_framing_max', ...
%     'MusNoise_up_l', 'MusNoise_up_m', 'MusNoise_up_u', 'MusNoise_up_max', ...
%     'MusNoise_down_l', 'MusNoise_down_m', 'MusNoise_down_u', 'MusNoise_down_max', ...
%     'MusNoise_to_ref_l', 'MusNoise_to_ref_m', 'MusNoise_to_ref_u', 'MusNoise_to_ref_max', ...
%     'MusNoise_to_test_l', 'MusNoise_to_test_m', 'MusNoise_to_test_u', 'MusNoise_to_test_max'};


OMOV = [MeanOS, MedianOS, ...
    MeanOS_RAW, MedianOS_RAW, ...
    TSM, ...
    MOV.WinModDiff1B, MOV.AvgModDiff1B, MOV.AvgModDiff2B, ...
    MOV.RmsNoiseLoudB, ...
    MOV.BandwidthRefB, MOV.BandwidthTestB, MOV.BandwidthTestB_new, ...
    MOV.TotalNMRB, ...
    MOV.RelDistFramesB, ...
    MOV.MFPDB, MOV.ADBB, ...
    MOV.EHSB, ...
    MOV.RmsModDiffA, MOV.RmsNoiseLoudAsymA, MOV.AvgLinDistA, MOV.SegmentalNMRB, ...
    MOV.DM, MOV.SER, ...
    MOV.peak_delta, MOV.transient_ratio, MOV.hpsep_transient_ratio, ...
    MOV.ave_P_ave_f, MOV.std_P_ave_f, ...
    MOV.ave_P_ave_f_mag, MOV.std_P_ave_f_mag, ...
    MOV.SS_MAD, MOV.SS_MD];


OMOV_name = {'MeanOS', 'MedianOS', ...
    'MeanOS_RAW', 'MedianOS_RAW', ...
    'TSM', ...
    'WinModDiff1B', 'AvgModDiff1B', 'AvgModDiff2B', ...
    'RmsNoiseLoudB', ...
    'BandwidthRefB', 'BandwidthTestB', 'BandwidthTestB_new', ...
    'TotalNMRB', ...
    'RelDistFramesB', ...
    'MFPDB', 'ADBB', ...
    'EHSB', ...
    'RmsModDiffA', 'RmsNoiseLoudAsymA', 'AvgLinDistA', 'SegmentalNMRB', ...
    'DM', 'SER', ...
    'peak_delta', 'transient_ratio', 'hpsep_transient_ratio', ...
    'MPhNW', 'SPhNW', ...
    'MPhMW', 'SPhMW', ...
    'SS_MAD','SS_MD'};



% % Save the MOV values for current file
f = split(test,'/');
fname = char(f(end));
fname = fname(1:end-4);
output_name = ['Features/' match_method '/' fname '.mat'];
save(output_name, 'OMOV');

end

