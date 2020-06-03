function [ bark ] = hz2bark( hz )
%[ bark ] = hz2bark( hz )
%   Converts a scalar or vector from Hz to Bark
%   Based on Eq 10 ITU-R BS.1387-1
bark = 7*asinh(hz/650);
end

