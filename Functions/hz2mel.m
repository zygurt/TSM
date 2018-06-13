function [ mel_f ] = hz2mel( hz_f )
%[ mel_f ] = hz2mel( hz_f )
%   Converts frequency in Hz to frequency in mel

mel_f = 2595*log10(1 + (hz_f./700)); % Hz scale to mel scale.

end

