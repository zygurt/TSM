function [ x ] = fb_outer_middle_ear( x )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global debug_var

if debug_var
    disp('  Filter Bank Outer and Middle Ear');
end
const = fb_constants();

fc_kHz = const.fc/1000;

W = -0.6*3.64*fc_kHz.^(-0.8) + ...
    6.5*exp(-0.6*(fc_kHz-3.3).^2) - ...
    (10^-3)*fc_kHz.^3.6;

Wt = repmat(10.^(W'/20),1,size(x,2));
x(:,:,1) = x(:,:,1) .* Wt;
x(:,:,2) = x(:,:,2) .* Wt;

end

