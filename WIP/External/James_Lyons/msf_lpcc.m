%% msf_lpcc - Linear Prediction Cepstral Coefficients
%
%   function feat = msf_lpcc(speech,fs,varargin)
%
% given a speech signal, splits it into frames and computes Linear Prediction Cepstral Coefficients for each frame.
%
% * |speech| - the input speech signal, vector of speech samples
% * |fs| - the sample rate of 'speech', integer
%
% optional arguments supported include the following 'name', value pairs 
% from the 3rd argument on:
%
% * |'winlen'| - length of window in seconds. Default: 0.025 (25 milliseconds)
% * |'winstep'| - step between successive windows in seconds. Default: 0.01 (10 milliseconds)
% * |'order'| - the number of coefficients to return. Default: 12
%
% Example usage:
%
%   lpccs = msf_lpcc(signal,16000,'order',10);
%
function feat = msf_lpcc(speech,fs,varargin)
    p = inputParser;   
    addOptional(p,'winlen',      0.025,@(x)gt(x,0));
    addOptional(p,'winstep',     0.01, @(x)gt(x,0));
    addOptional(p,'order',       12,   @(x)ge(x,1));
    addOptional(p,'preemph',     0,    @(x)ge(x,0));
    parse(p,varargin{:});
    in = p.Results;

    frames = msf_framesig(speech,in.winlen*fs,in.winstep*fs,@(x)hamming(x));
    temp = lpc(frames',in.order);
    temp = temp(:,2:end); % ignore leading ones
    feat = cepst(temp);

end

function ccs = cepst(apks)
% ccs = cepst(apks)
% - calculates cepstral coefficients from lpcs
% - apks are the lpc values (without leading 1)
%    - if more than one, apks should be a N by D matrix, where N is the
%    number of lpc vectors, D is the number of lpcs
% - ccs are the cepstral coefficients
% the number of ccs is the same as the number of lpcs
[N P] = size(apks);
ccs = zeros(N,P);

for i = 1:N
    for m = 1:P
        s = 0;
        for k = 1:(m-1)
            s = s + -1*(m - k)*ccs(i,m - k)*apks(i,k);
        end
        ccs(i,m) = -1*apks(i,m) + (1/m)*s;
    end
end
end
