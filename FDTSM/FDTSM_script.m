%Frequency Dependent Time Scale Modification Phase Vocoder
%This script demonstrates application of the FDTSM function.
%Each region is scaled independently.

close all
clear all
clc

pathInput = 'AudioIn/';
pathOutput = 'AudioOut/';
audio_file = 'Electropop.wav';

%Load audio file
[input, FS] = audioread([pathInput audio_file]);

%Sum to mono
num_chan = size(input,2);
if (num_chan == 2)
    input = sum(input,2);
    num_chan = size(input,2);
end

%Calculate the region parameters
N = 2048;
region.TSM = [1 0.5];
region.upper = [N/8 N/2];

%Frequency Dependent Time Scale Modification
y = FDTSM( input, N, region );

%Create the output name
output_filename = [audio_file(1:end-4) '_' sprintf('%.2f_',region.TSM) 'FDTSM.wav'];
%Save audio file
audiowrite(['AudioOut/' output_filename], y, FS);
