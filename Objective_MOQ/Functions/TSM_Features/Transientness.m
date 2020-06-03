function [ peak_delta, transient_ratio, hpsep_transient_ratio ] = Transientness( ref, test, General )
%UNTITLED2 Summary of this function goes here
%   Onset detection is based on Bello, JP., et. al, "A Tutorial on Onset
%   Detection in Music Signals", 2005
%   HP_seperation is abstracted from Driedger's TSM MATLAB Toolbox.
global debug_var

if debug_var
    fprintf('Transient Based Features\n');
end
if size(ref,2)>1
    ref = sum(ref,2);
    ref = ref/max(abs(ref));
else
    ref = ref/max(abs(ref));
end


[ ~, xPerc_ref ] = HP_seperation( ref );
[ ~, xPerc_test ] = HP_seperation( test );

ref_Perc_rms = rms(xPerc_ref);
test_Perc_rms = rms(xPerc_test);

hpsep_transient_ratio = ref_Perc_rms/test_Perc_rms;


ms = 16;
N = 2^nextpow2(General.fs*ms*10^(-3));
local_region = 2;
%Reference Peaks
ref_buf = buffer(ref,N,0.75*N);

w = repmat(0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))),1,size(ref_buf,2));
ref_buf_w = ref_buf.*w;

REF = fft(ref_buf_w,N);
REF_mag = abs(REF);
W = (1:N).^2;
W = repmat(W',1,size(REF_mag,2));
E_tilda_part = W.*REF_mag.^2;
E_tilda = sum(E_tilda_part,1);
E_log_tilda_diff = log10(E_tilda(2:end))-log10(E_tilda(1:end-1));
E_log_tilda_diff = E_log_tilda_diff(isfinite(E_log_tilda_diff));
p_ref_full = local_peak( E_log_tilda_diff, local_region,1);
p_ref = p_ref_full(E_log_tilda_diff(p_ref_full)>(mean(E_log_tilda_diff)+std(E_log_tilda_diff)));
num_ref_peaks = length(p_ref);

%Test Peaks
test_buf = buffer(test,N,0.75*N);
w = repmat(0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))),1,size(test_buf,2));
test_buf_w = test_buf.*w;
TEST = fft(test_buf_w,N);

TEST_mag = abs(TEST);
W = (1:N).^2;
W = repmat(W',1,size(TEST_mag,2));
E_tilda_part_TEST = W.*TEST_mag.^2;
E_tilda_TEST = sum(E_tilda_part_TEST,1);
E_log_tilda_diff_TEST = log10(E_tilda_TEST(2:end))-log10(E_tilda_TEST(1:end-1));
E_log_tilda_diff_TEST = E_log_tilda_diff_TEST(isfinite(E_log_tilda_diff_TEST));
%Find local peaks
p_test_full = local_peak( E_log_tilda_diff_TEST, local_region,1);
% p_test = E_log_tilda_diff_TEST(E_log_tilda_diff_TEST>(mean(E_log_tilda_diff_TEST)+std(E_log_tilda_diff_TEST)));
p_test = p_test_full(E_log_tilda_diff_TEST(p_test_full)>(mean(E_log_tilda_diff_TEST)+std(E_log_tilda_diff_TEST)));

num_test_peaks = length(p_test);

peak_delta = (num_test_peaks - num_ref_peaks)/(length(ref)/General.fs);

transient_ratio = mean(E_log_tilda_diff(p_ref))/mean(E_log_tilda_diff_TEST(p_test));

%     figure
% plot(ref)
% title('Reference')
% xlabel('Time(Samples)')
% 
% 
% N_test = 2*8192;
% fft_max_freq = 500;
% sample_start = 24000;
% w = 0.5*(1 - cos(2*pi*(0:N_test-1)'/(N_test-1)));
% FFT_mag_ref = log10(abs(fft(w.*ref(sample_start:sample_start+N_test-1))));
% FFT_mag_test = log10(abs(fft(w.*test(floor(sample_start/General.TSM):floor(sample_start/General.TSM)+N_test-1))));
% figure
% subplot(411)
% plot(ref(sample_start:sample_start+N_test-1))
% title('Reference Frame')
% xlabel('Time(Samples)')
% subplot(412)
% plot(linspace(0,General.fs/2,length(FFT_mag_ref(1:end/2))),FFT_mag_ref(1:end/2));
% axis([0 fft_max_freq 1.1*min(FFT_mag_ref(1:fft_max_freq)) 1.1*max(FFT_mag_ref(1:fft_max_freq))])
% title('Cropped Magnitude Spectrum of Reference Frame')
% xlabel('Frequency(Hz)')
% subplot(413)
% plot(linspace(0,General.fs/2,length(FFT_mag_test(1:end/2))),FFT_mag_test(1:end/2));
% axis([0 fft_max_freq 1.1*min(FFT_mag_test(1:fft_max_freq)) 1.1*max(FFT_mag_test(1:fft_max_freq))])
% title('Cropped Magnitude Spectrum of Test Frame')
% xlabel('Frequency(Hz)')
% subplot(414)
% plot(test(floor(sample_start/General.TSM):floor(sample_start/General.TSM)+N_test-1))
% title('Test Frame')
% xlabel('Time(Samples)')

if transient_ratio > 10
    disp('BIG Transient ratio.')
%     figure
%     subplot(211)
%     plot((1:length(ref))/General.fs,ref);
%     hold on
%     for n = 1:length(p_ref)
%         line([],[],'r')
%     end
%     
%     subplot(212)
%     plot((1:length(test))/General.fs,test);
%     hold on
%     plot((0.25*N*E_log_tilda_diff_TEST)/General.fs,E_log_tilda_diff_TEST)
%     hold off

end


if debug_var
    if peak_delta >= 0
        fprintf('For %s, there are %d more peaks\n',General.Testname, peak_delta);
    else
        fprintf('For %s, there are %d less peaks\n',General.Testname, -1*peak_delta);
    end
    
    
    
    
%     if isnan(transient_ratio)
%         p_test = local_peak( E_log_tilda_diff_TEST, local_region);
%         E_log_tilda_diff_TEST = E_log_tilda_diff_TEST(isfinite(E_log_tilda_diff_TEST));
%         plot(E_log_tilda_diff_TEST)
%         hold on
%         line([0 length(E_log_tilda_diff_TEST)],[mean(E_log_tilda_diff_TEST)+std(E_log_tilda_diff_TEST) mean(E_log_tilda_diff_TEST)+std(E_log_tilda_diff_TEST)])
%         hold off
%         p_test = p_test(E_log_tilda_diff_TEST(p_test)>(mean(E_log_tilda_diff_TEST)+std(E_log_tilda_diff_TEST)));
%         
%         
%         fprintf('Nan')
%     end
    
end

% figure
% plot(1:length(E_log_tilda_diff), E_log_tilda_diff)
% hold on
% plot((1:length(E_log_tilda_diff_TEST))*General.TSM,E_log_tilda_diff_TEST)
% hold off


end

