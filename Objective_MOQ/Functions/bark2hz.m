function [ hz ] = bark2hz( bark )
%[ bark ] = hz2bark( hz )
%   Converts a scalar or vector from Hz to Bark
%   Based on Thiede 2000, PEAQ - The ITO Standard for Objective Measure of
%   Perceived Audio Quality
%   Equation 2
hz = 650*sinh(bark/7);
end

