function [ MOV ] = RelDistFramesB( Model_Ref, Model_Test, Pro_Test )
%[ MOV ] = RelDistFramesB( Model_Ref, Model_Test, Pro_Test )
%   Calculation of Relative Disturbed Frames
%   As described by ITU-R BS.1387-1 Section 4.6
global debug_var

if debug_var
    disp('    RelDistFramesB')
end
N = size(Pro_Test.P_noise,1);
temp = zeros(N,1);
for n = 1:size(Pro_Test.P_noise,1)
    temp(n) = max(10*log10(Pro_Test.P_noise(n,:)./Model_Test.M(n,:)));
end

%Ignore low energy frames
ref_start = Model_Ref.ref_start;
ref_end = Model_Ref.ref_end;
frame_starts = 1024*(0:N-1)+1;
frame_ends = 1024*(0:N-1)+2048;

start_frame = length(frame_starts(frame_starts<ref_start))+1;
end_frame = N-length(frame_ends(frame_ends>ref_end));
%Calculate the MOV.  ITU-R BS.1387-1 says 'related to the total number of
%frames'  We have taken that to mean ratio. This is confirmed by Kabal in
%"An Examination and Interpretation of ITU-R BS.1387: Perceptual Evaluation of Audio Quality"
MOV = sum(temp(start_frame:end_frame)>1.5)/N; 


end
