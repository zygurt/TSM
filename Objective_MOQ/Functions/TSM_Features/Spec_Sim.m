function [SS_MAD, SS_MD] = Spec_Sim(sig_ref, sig_test)
%[SS_MAD, SS_MD] = Spec_Sim(sig_ref, sig_test)
%   Returns a measure of Spectral similarity
global debug_var

if debug_var
    disp('Spectral Similarity');
end
N = 2048;
Ha = N/4;
ref_len = size(sig_ref,1);
test_len = size(sig_test,1);
len_ratio = ref_len/test_len;
ref_frame_starts = 1:Ha:ref_len-N;
test_frame_starts = ceil(ref_frame_starts/len_ratio);
%Buffer the input signals
ref_buf = vec_buffer(sig_ref, N, ref_frame_starts);
test_buf = vec_buffer(sig_test, N, test_frame_starts);
%Transform to Frequency Domain
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))); %Hann
wn = repmat(w,1,size(ref_buf,2));
REF_BUF = fft(wn.*ref_buf,N);
REF_BUF_MAG = abs(REF_BUF(1:N/2+1,:));
REF_BUF_MAG_NORM = REF_BUF_MAG/max(max(REF_BUF_MAG));

TEST_BUF = fft(wn.*test_buf,N);
TEST_BUF_MAG = abs(TEST_BUF(1:N/2+1,:));
TEST_BUF_MAG_NORM = TEST_BUF_MAG/max(max(TEST_BUF_MAG));
TEST_BUF_ANGLE = angle(TEST_BUF(1:N/2+1,:));
%Pre-Allocate
SS_vec_MD = zeros(1,size(TEST_BUF_ANGLE,2));
SS_vec_MAD = zeros(1,size(TEST_BUF_ANGLE,2));
%For every frame
for u = 1:size(TEST_BUF_ANGLE,2)
    x = (1:1025)';
    %Fit a 3rd order polynomial to Magnitude spectrum in dB
    P_ref = polyfit(x,10*log10(REF_BUF_MAG_NORM(:,u)),3);
    %Create signal using coefficients, excluding intercept
    yfit_ref = P_ref(1)*x.^3+P_ref(2)*x.^2+P_ref(3)*x+P_ref(4);
    %Fit a 3rd order polynomial to Magnitude spectrum in dB
    P_test = polyfit(x,10*log10(TEST_BUF_MAG_NORM(:,u)),3);
    %Create signal using coefficients, excluding intercept
    yfit_test = P_test(1)*x.^3+P_test(2)*x.^2+P_test(3)*x+P_test(4);
    %Compute the difference between the polynomial signals for each frame
    SS_vec_MD(u) = mean(yfit_ref-yfit_test);
    SS_vec_MAD(u) = mean(abs(yfit_ref-yfit_test));
    
%     if isnan(SS_vec_MD(u))
%         disp('Nana')
%     end
    
end
%Average all of the differences
SS_MD = mean(SS_vec_MD);
SS_MAD = mean(abs(SS_vec_MAD));
% if isnan(SS)
%     disp('Nana')
% end



end


