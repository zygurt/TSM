function [H, bl, bh] = melfbank(M, N, fs)
% MELFBANK - creates triangular mel filter banks.
%
% Inputs:
%   M - number of filterbanks.
%   N - is the length of each filter (N/2 + 1 typically).
%   fs - sampling frequency.
%
% Outputs:
%   H - triangular mel filterbank matrix.
%	  bl - lower boundary point frequencies.
%	  hl - higher boundary point frequencies.
%
%% FILE:           melfbank.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Computes triangular mel filter banks.
%% REFERENCE: 
%	Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing: 
% A guide to theory, algorithm, and system development. 
% Prentice Hall, Upper Saddle River, NJ, USA (pp. 315).

  fl = 0; % lowest frequency (Hz).
  fh = fs/2; % highest frequency (Hz).
  H = zeros(M, N); % mel filter bank.
  bl = zeros(1, M); % lower boundary point bin numbers.
  bh = zeros(1, M); % higher boundary point bin numbers.
  for m = 1:M
    bl(m) = bpoint(m - 1, M, N, fs, fl, fh); % lower boundary point, f(m - 1) for m-th filterbank.
    c = bpoint(m, M, N, fs, fl, fh); % m-th filterbank centre point, f(m).
    bh(m) = bpoint(m + 1, M, N, fs, fl, fh); % higher boundary point f(m + 1) for m-th filterbank.
    for k = 0:N-1
      if k >= bl(m) && k <= c
        H(m,k+1)=(k - bl(m))/(c - bl(m)); % m-th filterbank up-slope. 
      end
      if k >= c && k <= bh(m)
        H(m,k+1)=(bh(m) - k)/(bh(m) - c); % m-th filterbank down-slope. 
      end
    end
  end
  bl = (bl*(fs/2)/N); % convert lower boundary points from bin number to frequency.
  bh = (bh*(fs/2)/N); % convert higher boundary points from bin number to frequency.
end