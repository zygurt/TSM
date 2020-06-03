function [ X_dBA ] = dBA( X, fs )
%[ X_dBA ] = dBA( X, fs )
%   Conversion of Magnitude Spectrograms to dBA
%  IEC 61672-1:2013 Electroacoustics - Sound level meters - Part 1: Specifications. IEC. 2013

f = [0 (1:(size(X,1)-1))/(size(X,1)-1)*fs/2];

R_A = (12194^2*f.^4./((f.^2+20.6^2).*sqrt((f.^2+107.7^2).*(f.^2+737.9^2)).*(f.^2+12194^2))).';
% semilogx(f,R_A)

% [~,b_1000] = min(abs(f-1000));

X_dBA = X.*repmat(R_A,1,size(X,2));

end

