function [ H ] = filterbank( c, N )
%[ H ] = mel_filterbank( K, N, fs )
%   Creates mel spaced filterbanks from 0:high(Hz) and 1:N/2+1
% K = number of filters
% high = highest frequency to create filter banks up to
% N = length of FFT
% fs = sample rate


%Calculate the lower bounds of each region
% lower = [1 upper(1:end-1)+1];
% num_regions = length(lower);
% c = round((lower+upper)/2); %Centre of each region
c = round(c);
K = length(c);
H = zeros(K,N/2+1);

%Initial band
H(1,1:c(1)) = linspace(0,1,c(1));
H(1,c(1):c(2)) = linspace(1,0,c(2)-c(1)+1);
%Middle Bands
for k = 2:K-1
    %Up slope
    H(k,c(k-1):c(k)) = linspace(0,1,c(k)-c(k-1)+1);
    %Down slope
    H(k,c(k):c(k+1)) = linspace(1,0,c(k+1)-c(k)+1);
end
%Final band
H(K,c(K-1):c(K)) = linspace(0,1,c(K)-c(K-1)+1);
H(K,c(K):N/2+1) = linspace(1,0,N/2+1-c(K)+1);

end


