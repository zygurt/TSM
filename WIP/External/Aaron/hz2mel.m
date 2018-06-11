function m = hz2mel(f)
% HZ2MEL - converts Hz scale to mel scale.
%
% Input:
%	f - Hz value.
%
% Output:
%	m - mel value.

%% FILE:           hz2mel.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Converts Hz scale to mel scale.
  m = 2595*log10(1 + (f/700)); % Hz scale to mel scale.
end