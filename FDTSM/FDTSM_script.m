%Frequency Dependent Time Scale Modification Phase Vocoder
%This script demonstrates application of the FDTSM function.
%Each region is scaled independently.

close all
clear all
clc

pathInput = 'AudioIn/';
pathOutput = 'AudioOut/';
audio_file = 'Male_Speech.wav';

%Load audio file
[x, FS] = audioread([pathInput audio_file]);

%Calculate the region parameters
N = 2048;	% Length of the frame
region.TSM = [0.5 1];	% The TSM ratio is the speed of playback
			% 0.5 = 50%, 1 = 100% and 2 = 200%
region.upper = [N/16 N/2+1];	%Length of region.TSM and region.upper must be equal
				%Final value in the vector must be N/2+1


%Frequency Dependent Time Scale Modification
y = FDTSM( x, N, region );

%Create the output name
output_filename = [audio_file(1:end-4) '_' sprintf('%.2f_',region.TSM) 'FDTSM.wav'];
if length(output_filename)>127
    output_filename = [audio_file(1:end-4) '_processed_FDTSM.wav'];
end
%Save audio file
audiowrite([pathOutput output_filename], y, FS);









