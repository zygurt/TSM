function [ Frame_mid, File_mid ] = st_balance( input, N, norm )
%Calculate the frame and file stereo midpoint
%   input is a 2 channel signal
%   N is the frame length. Larger values give smoother output.
%   norm is a flag to normalise output -1->1 for R->L for a sine wave
%   Frame_mid is the average midpoint for each frame in the input signal
%   File_mid is the average midpoint for the input signal

% Tim Roberts - Griffith University 2018

if max(max(abs(input(:,:)))) ~= 0
    input_l=abs(input(:,1))./max(max(abs(input(:,:))));
    input_r=abs(input(:,2))./max(max(abs(input(:,:))));
    if(norm)
        res = (input_l-input_r)*1.5709;
    else
        res = (input_l-input_r);
    end
    Frame_mid = mean(buffer(res,N));
    File_mid = mean(Frame_mid);
else
    %Signal is silence
    Frame_mid = 0;
    File_mid = 0;
end

end

