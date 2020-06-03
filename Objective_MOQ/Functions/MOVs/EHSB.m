function [ MOV ] = EHSB( Pro_Test, General )
%[ MOV ] = EHSB( Pro_Test, General )
%   As described by ITU-R BS.1387-1 Section 4.8.1
global debug_var

if debug_var
disp('    EHSB');
end
% save('EHSB_test.mat');

fs = General.fs;
N = General.N;
f = 18000;

b = round((f/(fs/2))*(N/2));
max_lag = b/2;

lags = 2^(nextpow2(max_lag)-1);
%f0 is the error vector
%ft is the same vector lagged by a certain amount
% first_autocorr is ft(0) and f0(0)
% last_autocorr is ft(0) and f0(255)
F = Pro_Test.F_noise;
N = size(Pro_Test.F_noise,1);
C = zeros(size(F,1),lags);
for n = 1:N
    for lag = 0:lags-1
%         auto_corr(n,lag) = sum(F(n,1:lags)*F(n,lags:end));
%         C(n,lag+1) = cos(dot(F(n,1:end-lag),F(n,lag+1:end)')/dot(abs(F(n,1:end-lag)),abs(F(n,lag+1:end))));
        C(n,lag+1) = dot(F(n,1:end-lag),F(n,lag+1:end))/(sqrt(sum(F(n,1:end-lag).^2))*sqrt(sum(F(n,lag+1:end).^2)));

%         subplot(211)
%         plot(F(n,1:end-lag));
%         subplot(212)
%         plot(F(n,lag+1:end));
    end
end

w = PEAQ_Hann(lags);
w = w/max(w);

auto_corr_w = C.*repmat(w,size(C,1),1);
auto_corr_nodc = auto_corr_w-mean(auto_corr_w,2);
% AUTO_CORR = fft(auto_corr_w');
AUTO_CORR = fft(auto_corr_nodc');
AUTO_CORR_pow = abs(AUTO_CORR(1:lags/2+1,:)).^2;
AUTO_CORR_pow(isnan(AUTO_CORR_pow)) = 0; %Added this to remove NaN values that crept in for 1 file.
%This happens if there is no difference in Outer ear weighted FFT outputs
%for every frame of a bin. This gives 0's in error_sig.m, which are then
%used in the C calculation above, resulting in deviding by 0.
peaks = first_peaks(AUTO_CORR_pow);


% peaks = find_peaks(AUTO_CORR_pow);
% pos = 1;
% first_peak = zeros(length(peaks),1);
% for n = 1:length(peaks)
%     if (~isempty(peaks(n).pa))
%         first_peak(pos) = min(peaks(n).pa);
%         pos = pos+1;
%     end
% end
% first_peak = first_peak(1:pos-1);

MOV = PEAQ_Temporal_Average(peaks,'Linear');

end

