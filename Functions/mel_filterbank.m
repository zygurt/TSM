function [ H ] = mel_filterbank( K, high, N, fs )
%[ H ] = mel_filterbank( K, N, fs )
%   Creates mel spaced filterbanks from 0:high(Hz) and 1:N/2+1
% K = number of filters
% high = highest frequency to create filter banks up to
% N = length of FFT
% fs = sample rate

low_mel = 0;
high_mel = hz2mel(high);

cp_mel = linspace(low_mel,high_mel,K+2); %filter centers
cp_mel = cp_mel(2:end-1);
%convert mel points to Hz fft bin numbers
bin_width = (fs/2)/(N/2+1);
cp_hz = mel2hz(cp_mel);
cp_bins = floor(cp_hz/bin_width)+1;  %+1 for matlab indexing
top_bin = floor(high/bin_width);
H = zeros(K,N/2+1);
%Initial band
H(1,1:cp_bins(1)) = linspace(0,1,cp_bins(1));
H(1,cp_bins(1):cp_bins(2)) = linspace(1,0,cp_bins(2)-cp_bins(1)+1);
%Middle Bands
for k = 2:K-1
    %Up slope
    H(k,cp_bins(k-1):cp_bins(k)) = linspace(0,1,cp_bins(k)-cp_bins(k-1)+1);
    %Down slope
    H(k,cp_bins(k):cp_bins(k+1)) = linspace(1,0,cp_bins(k+1)-cp_bins(k)+1);
end
%Final band
H(K,cp_bins(K-1):cp_bins(K)) = linspace(0,1,cp_bins(K)-cp_bins(K-1)+1);
H(K,cp_bins(K):top_bin) = linspace(1,0,top_bin-cp_bins(K)+1);

end

