function [ X_dBA ] = dBA_Torcolli( X, fs )
%[ X_dBA ] = dBA_Torcolli( X, fs )
%   Convert Spectrogram to dBA
%   Code from Matteo Torcolli

%A-weighting filter coefficients

c1 = 3.5041384e16;
c2 = 20.598997^2;
c3 = 107.65265^2;
c4 = 737.86223^2;
c5 = 12194.217^2;
 
nbins = size(X,1);

% evaluate the A-weighting filter in the frequency domain
f2 = (fs*0.5*(0:nbins-1)/nbins).^2 ;
num = c1*(f2.^4);
den = ((c2+f2).^2) .* (c3+f2) .* (c4+f2) .* ((c5+f2).^2);
Aw = num./den;
Aw = Aw(:);
 
% convert to dBA
X_dBA = 10*log10( Aw .* abs(X).^2) ;


end

