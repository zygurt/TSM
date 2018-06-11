%% msf_mfcc - Mel Frequency Cepstral Coefficients
%
%   function feat = msf_mfcc(speech,fs,varargin)
%
% given a speech signal, splits it into frames and computes Mel frequency cepstral coefficients for each frame.
% For a tutorial on MFCCs, see <http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/ MFCC tutorial>.
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
% * |'ncep'| - the number of cepstral coeffients to use. Default: 13
% * |'liftercoeff'| - liftering coefficient, 0 is no lifter. Default: 22
% * |'appendenergy'| - if true, replaces 0th cep coeff with log of total frame energy. Default: true
%
% Example usage:
%
%   mfccs = msf_mfcc(signal,16000,'nfilt',40,'ncep',12);
%
function mfccs = msf_mfcc(speech,fs,varargin)
    p = inputParser;   
    addOptional(p,'winlen',      0.025,@(x)gt(x,0));
    addOptional(p,'winstep',     0.01, @(x)gt(x,0));
    addOptional(p,'nfilt',       26,   @(x)ge(x,1));
    addOptional(p,'lowfreq',     0,    @(x)ge(x,0));
    addOptional(p,'highfreq',    fs/2, @(x)ge(x,0));
    addOptional(p,'nfft',        512,  @(x)gt(x,0));
    addOptional(p,'ncep',        13,   @(x)ge(x,1));          
    addOptional(p,'liftercoeff', 22,   @(x)ge(x,0));          
    addOptional(p,'appendenergy',true, @(x)ismember(x,[true,false]));          
    addOptional(p,'preemph',     0,    @(x)ge(x,0));    
    parse(p,varargin{:});
    in = p.Results;
    H = msf_filterbank(in.nfilt, fs, in.lowfreq, in.highfreq, in.nfft);
    pspec = msf_powspec(speech, fs, 'winlen', in.winlen, 'winstep', in.winstep, 'nfft', in.nfft);
    en = sum(pspec,2); % energy in each frame
    feat = dct(log(H*pspec'))';
    mfccs = lifter(feat(:,1:in.ncep), in.liftercoeff);
    if in.appendenergy
        mfccs(:,1) = log10(en);
    end
    
end

function lcep = lifter(cep,L)
    [N,D] = size(cep);
    n = 0:D-1;
    lift = 1 + (L/2)*sin(pi*n/L);
    lcep = cep .* repmat(lift,N,1);
end

