function [ hz_f ] = mel2hz( mel_f )
%[ hz_f ] = mel2hz( mel_f )
%   Converts frequency in mel to frequency in Hz

hz_f = 700*(10.^(mel_f./2595) -1);

end

