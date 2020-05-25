function [x_per] = inv_prctile(x,V)
%Find the closest percentile [0,100] to the given value
%   x is value
%   V is vector of known values
%   x_per is the closest percentile

if size(V,2)>size(V,1)
    V = V.';
end
p = 0:100;
P = prctile(V,p);
dP = P-x;
[~,x_per] = min(abs(dP));
end

