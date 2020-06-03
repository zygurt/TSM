function [ x_bank ] = fb_filter_bank( x, General )
%UNTITLED5 Summary of this function goes here
%   Implemented as per ITU-R BS.1387-1 Section 2.2.5
global debug_var

if debug_var
    disp('  Filter Bank Decomposition');
end

Z = 40;
T = 1/General.fs;

const = fb_constants();
h_re = zeros(Z,max(const.Nk));
h_im = zeros(size(h_re));
for k = 1:Z
    for n = 1:const.Nk(k)
        h_re(k,n) = (4./const.Nk(k)) * (sin((pi*n)/const.Nk(k))).^2 * cos(2*pi*const.fc(k)*(n-const.Nk(k)/2)*T);
        h_im(k,n) = (4./const.Nk(k)) * (sin((pi*n)/const.Nk(k))).^2 * sin(2*pi*const.fc(k)*(n-const.Nk(k)/2)*T);
    end
end
s = 1;
for k = 1:Z
    %Calculate the filter output for every 32nd sample
    %Add code for the additional delay
    for n = const.Nk(k):32:length(x)-(const.Nk(k)+const.Dk(k))
        x_bank(k,s,1) = h_re(k,1:const.Nk(k))*x(n-const.Nk(k)+1+const.Dk(k):n+const.Dk(k));
        x_bank(k,s,2) = h_im(k,1:const.Nk(k))*x(n-const.Nk(k)+1+const.Dk(k):n+const.Dk(k));
        s = s+1;
    end
    s = 1;
end


end

