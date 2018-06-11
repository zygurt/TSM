%% msf_rc - Reflection Coefficients
%
%   function feat = msf_rc(speech,fs,varargin)
%
% given a speech signal, splits it into frames and computes Reflection Coefficients for each frame.
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
%   rcs = msf_rc(signal,16000,'order',10);
%
function feat = msf_rc(speech,fs,varargin)
    p = inputParser;   
    addOptional(p,'winlen',      0.025,@(x)gt(x,0));
    addOptional(p,'winstep',     0.01, @(x)gt(x,0));
    addOptional(p,'order',       12,   @(x)ge(x,1));
    addOptional(p,'preemph',     0,    @(x)ge(x,0));
    parse(p,varargin{:});
    in = p.Results;

    frames = msf_framesig(speech,in.winlen*fs,in.winstep*fs,@(x)hamming(x));
    temp = lpc(frames',in.order);
    feat = zeros(size(temp,1),in.order);
    for i = 1:size(temp,1)
        feat(i,:) = poly2rc(temp(i,:))';
    end
    
end
