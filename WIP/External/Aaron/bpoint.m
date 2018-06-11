function f = bpoint(m, M, NFFT, fs, fl, fh)
% BPOINT - detirmines the frequency bin boundary point for a filterbank.
%
% Inputs:
%	m - filterbank.
%	M - total filterbanks.
%	NFFT - number of frequency bins.
%	fs - sampling frequency.
%	fl - lowest frequency.
%	fh - highest frequency.
%
% Output:
%	f - frequency bin boundary point.

%% FILE:           bpoint.m
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Detirmines the frequency bin boundary point for a filterbank.
  f = ((2*NFFT)/fs)*mel2hz(hz2mel(fl) + m*((hz2mel(fh) - hz2mel(fl))/(M + 1))); % boundary point.
end
