function [ MOV ] = TotalNMRB( Model_Ref, Model_Test, Pro_Test )
%[ MOV ] = TotalNMRB( Model_Ref, Model_Test, Pro_Test )
%   As described by ITU-R BS.1387-1 Section 4.5.1
global debug_var

if debug_var
disp('    TotalNMRB');
end
Z = size(Model_Test.M,2);
N = size(Model_Test.M,1);
NMR = zeros(N,1);
for n = 1:N
    NMR(n) = (1/Z)*sum(Pro_Test.P_noise(n,:)./Model_Test.M(n,:));
end

%Ignore low energy frames
ref_start = Model_Ref.ref_start;
ref_end = Model_Ref.ref_end;
frame_starts = 1024*(0:N-1)+1;
frame_ends = 1024*(0:N-1)+2048;

start_frame = length(frame_starts(frame_starts<ref_start))+1;
end_frame = N-length(frame_ends(frame_ends>ref_end));

MOV = 10*log10((1/N)*sum(NMR(start_frame:end_frame)));
end

