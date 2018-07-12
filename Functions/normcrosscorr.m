function [ Y_ncc ] = normcrosscorr( A, B )
%[ Y_ncc ] = normcrosscorr( A, B )
%Calculates the normalised cross correlation
%   A and B must be the same length

A_mean = mean(A);
B_mean = mean(B);
A_std_dev = std(A);
B_std_dev = std(B);
n = length(A);

Y_ncc = sum( ((A-A_mean).*(B-B_mean))/(A_std_dev*B_std_dev) )/n;

end

