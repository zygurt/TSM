function [x_per] = inv_prctile(x,V,d)
%Find the closest percentile [0,100] to the given value
%   x is value
%   V is vector of known values
%   d is the direction of improvement 'up' or 'down'
%   x_per is the closest percentile

if size(V,2)>size(V,1)
    V = V.';
end
switch d
    case 'up'
        p = 0:99;
    case 'down'
        p = 99:-1:0;
    otherwise
        fprintf('Direction unknown.  Using greater as improvement\n')
        p = 0:99;
end
P = prctile(V,p);
dP = P-x;
[~,x_per] = min(abs(dP));
end

