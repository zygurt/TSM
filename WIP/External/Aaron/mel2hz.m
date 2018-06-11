function f = mel2hz(m)
% MEL2HZ - converts mel scale to Hz scale.
%
% Input:
%	m - mel value.
%
% Output:
%	f - Hz value.

%% FILE:           mel2hz.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Converts mel scale to Hz scale.
  f = 700*((10^(m/2595)) - 1); % mel scale to Hz scale.
end