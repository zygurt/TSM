function [ Frame_SPC, File_SPC  ] = st_phase_coherence( x, N )
% [ Frame_SPC, File_SPC  ] = st_phase_coherence( x, N )
% Calculate the frame and file stereo phase coherence (SPC)
%   x is a 2 channel signal
%   N is the frame length.  Larger values give smoother output.
%   Frame_SPC is the SPC for each frame
%   File_SPC is the SPC for the entire file

% Tim Roberts - Griffith University 2018

input_l=x(:,1)./max(max(abs(x(:,:))));
input_r=x(:,2)./max(max(abs(x(:,:))));
coherence = input_l.*input_r;

coherence(coherence>0)=1;
coherence(coherence<0)=-1;

Y = buffer(coherence,N);
Frame_SPC = mean(Y);
File_SPC = mean(Frame_SPC);
end

