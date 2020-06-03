function [ w ] = PEAQ_Hann( N )
%[ w ] = PEAQ_Hann( N )
%   Generates the Hann window specified by
%   ITU-R BS.1387-1 Section 2.1.3
%   Equation (2)
w = 0.5*sqrt(8/3)*(1-cos(2*pi*(1:N)/(N-1)));

end

