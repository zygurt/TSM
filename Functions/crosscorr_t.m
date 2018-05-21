function [xc] = crosscorr_t(x,y)
% [xc] = crosscorr_t(x,y)
% Compute the normalised time domain cross correlation of two vectors.
% Vectors must be the same length

% Tim Roberts - Griffith University 2018

den = sqrt(sum(x.^2)*sum(y.^2));
if den == 0
    xc = 0;
else
    xc = sum(x.*y)/den;
end
end