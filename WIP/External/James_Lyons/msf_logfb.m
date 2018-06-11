%% msf_logfb - Log Filterbank Energies
%
%   function feat = msf_logfb(speech,fs,varargin)
%
% given a speech signal, splits it into frames and computes log filterbank energies for each frame.
%
% * |speech| - the input speech signal, vector of speech samples
% * |fs| - the sample rate of 'speech', integer
%
% optional arguments supported include the following 'name', value pairs 
% from the 3rd argument on:
%
% * |'winlen'| - length of window in seconds. Default: 0.025 (25 milliseconds)
% * |'winstep'| - step between successive windows in seconds. Default: 0.01 (10 milliseconds)
% * |'nfilt'| - the number filterbanks to use. Default: 26
% * |'lowfreq'| - the lowest filterbank edge. In Hz. Default: 0    
% * |'highfreq'| - the highest filterbank edge. In Hz. Default: fs/2
% * |'nfft'| - the FFT size to use. Default: 512
%
% Example usage:
%
%   logfbs = msf_logfb(signal,16000,'nfilt',40,'ncep',12);
%
function feat = msf_logfb(speech,fs,varargin)
    p = inputParser;   
    addOptional(p,'winlen',      0.025,@(x)gt(x,0));
    addOptional(p,'winstep',     0.01, @(x)gt(x,0));
    addOptional(p,'nfilt',       26,   @(x)ge(x,1));
    addOptional(p,'lowfreq',     0,    @(x)ge(x,0));
    addOptional(p,'highfreq',    fs/2, @(x)ge(x,0));
    addOptional(p,'nfft',        512,  @(x)gt(x,0));
    addOptional(p,'preemph',     0,    @(x)ge(x,0));    
    parse(p,varargin{:});
    in = p.Results;
    H = msf_filterbank(in.nfilt,fs,in.lowfreq,in.highfreq,in.nfft);
    pspec = msf_powspec(speech,fs,'winlen',in.winlen,'winstep',in.winstep,'nfft',in.nfft);
    feat = log(pspec*H');
end
