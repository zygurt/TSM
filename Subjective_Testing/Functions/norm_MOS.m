function [ Zi ] = norm_MOS( xi, xsi, xs, ss, ssi )
%[ Zi ] = norm_MOS( xi, xsi, xs, ss, ssi )
%   As per ITU-R BS.1284-1
%   xi = MOS score
%   xsi = mean MOS for subject in session
%   xs = mean of ALL subjects in session
%   ss = standard deviation for ALL subjects in session
%   ssi = standard deviation for subject in session
Zi = ((xi-xsi)/ssi)*ss+xs;

end

