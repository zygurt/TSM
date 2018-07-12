function [ Frame_mid, File_mid ] = st_balance( x, N )
% [ Frame_mid, File_mid ] = st_balance( x, N )
% Calculate the frame and file stereo midpoint
%   x is a 2 channel signal
%   N is the frame length. Larger values give smoother output.
%   Frame_mid is the average midpoint for each frame in the input signal
%   File_mid is the average midpoint for the input signal

% Tim Roberts - Griffith University 2018

if max(max(abs(x(:,:)))) ~= 0
    input_l=abs(x(:,1))./max(max(abs(x(:,:))));
    input_r=abs(x(:,2))./max(max(abs(x(:,:))));
    res = (input_l-input_r);
    Frame_mid = mean(buffer(res,N));
    File_mid = mean(Frame_mid);
else
    %Signal is silence
    Frame_mid = 0;
    File_mid = 0;
end

end

