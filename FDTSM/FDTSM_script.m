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
[input, FS] = audioread([pathInput audio_file]);

%Calculate the region parameters
% N = 2048;	% Length of the frame
% region.TSM = [0.5 1];	% The TSM ratio is the speed of playback
% 			% 0.5 = 50%, 1 = 100% and 2 = 200%
% region.upper = [N/16 N/2+1];	%Length of region.TSM and region.upper must be equal
% 				%Final value in the vector must be N/2+1

%Figure 3
% N = 2048;
% num_regions = N/2+1;
% region.TSM = linspace(0.5,2,num_regions);
% region.upper = 1:(N/2+1);

%Figure 4
% N = 2048;
% num_regions = N/2+1;
% low=0.5;
% high=2;
% region.TSM = (low + (high-low).*rand(num_regions,1))';
% slow_bin = 50;
% region.TSM(slow_bin) = 0.2;
% region.upper = 1:(N/2+1);

%Figure 5
% N = 2048;
% num_regions = 32;
% low=0.1;
% high=1;
% region.TSM = (low + (high-low).*rand(num_regions,1))';
% region.upper = ceil(linspace(1,N/2+1,num_regions));

%Figure 6
%Use FDTSM_GUI_example script

%Figure 7
N = 2048;
region.TSM = [1 0.9 1];
region.upper = [N/256 N/16 N/2+1];

%Figure 8
% pathInput = 'AudioOut/';
% audio_file = 'Figure_3.wav';
% [input, FS] = audioread([pathInput audio_file]);
% N = 2048;
% num_regions = N/2+1;
% region.TSM = 1./linspace(0.5,2,num_regions);
% region.upper = 1:(N/2+1);


%Frequency Dependent Time Scale Modification
y = FDTSM( input, N, region );

%Create the output name
output_filename = [audio_file(1:end-4) '_' sprintf('%.2f_',region.TSM) 'FDTSM.wav'];
if length(output_filename)>127
    output_filename = [audio_file(1:end-4) '_processed_FDTSM.wav'];
end
%Save audio file
audiowrite([pathOutput output_filename], y, FS);

