function [ peaks ] = find_peaks_log( mag_X )
% [ peaks ] = find_peaks_log( mag_X )
% Find the peaks of an array.  Takes the magnitude spectrum as input
%   The function finds the peaks of an array and returns the peak location
%   An element is considered a peak as per the table below

%   Bins 1-16 = peaks
%   Bins 17-32 peak if greater than 2 neighbours
%   Bins 33-64 peak if greater than 4 neighbours
%   Bins 65-128 peak if greater than 8 neighbours
%   Bins 129-256 peak if greater than 16 neighbours
%   All remaining bins peak if greater than 32 neighbours

%   As per Karrer et al 'PhaVoRIT: A Phase Vocoder for Real-Time
%   Interactive Time-Stretching' 2006
%   This function also returns the lower and upper bounds of the regions of
%   each peak
%   This function expects column magnitude data and returns peaks for each
%   column
% peaks(c).pa  => Peak Array
% peaks(c).rl  => Region Lower Bound
% peaks(c).ru  => Region Upper Bound
% peaks(c).empty_flag  => 1 = empty, 0 = data

% Tim Roberts - Griffith University 2018

num_chan = size(mag_X,2);
%Initialise
pad_mag_X = [mag_X; zeros(32,num_chan)];
for c = 1:num_chan
    peaks(c).pa = (1:16)';
    peaks(c).empty_flag = 0; %There will always be peaks.
end


each_side = [1 2 4 8 16]; %Bins each side to check
transition = [32 64 128 256 (length(pad_mag_X)+1)];


%Find the peaks
for c = 1:num_chan
    index = 1;
    cont = 1;
    for k = 17:length(mag_X)
        if k<transition(index)
            for p = -each_side(index):each_side(index)
                if pad_mag_X(k,c)<pad_mag_X(k+p,c) && cont == 1
                    cont = 0;
                end
            end
            if cont == 1
                peaks(c).pa = [peaks(c).pa; k];
            else
                cont = 1;
            end
        else
            index = index+1;
            k = k-1;    %deliberate to reset back into checking bins
        end
    end
end
%Set the region bounds as half way between peaks
for c = 1:num_chan
    if (peaks(c).empty_flag)
        %There are no peaks
        peaks(c).rl = [];
        peaks(c).ru = [];
    else
        %Take 0 padding into account when calculating the upper and lower
        %regions.
        peaks(c).rl = zeros(length(peaks(c).pa),1);
        peaks(c).ru = zeros(length(peaks(c).pa),1);
        peaks(c).rl(1:16) = 1:16;
        peaks(c).ru(1:16) = 1:16;
        
        peaks(c).rl(17:end) = ceil((peaks(c).pa(17:end)+peaks(c).pa(16:end-1))/2);
        peaks(c).ru(16:end-1) = peaks(c).rl(17:end) - 1;
        peaks(c).ru(length(peaks(c).ru)) = length(mag_X);
        
    end
end
end