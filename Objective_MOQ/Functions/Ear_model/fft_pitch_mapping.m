function [ P ] = fft_pitch_mapping( Fe, fs, bands )
%[ P ] = pitch_mapping( Fsp, fs, N, bands )
%   ITU-R BS.1387-1 Section 2.1.5.1
%   Implements the Pseudocode
global debug_var

if debug_var
disp('  Pitch Mapping')
end
Fsp = abs(Fe).^2;
N = (size(Fsp,2)-1)*2;

Fres = fs/N;

Z = length(bands);
fl = bands(1,:);
fu = bands(3,:);
P = zeros(size(Fsp,1),Z);
for n = 1:size(Fsp,1)
    for i=1:Z
        for k = 1:N/2+1
            if ((k-0.5)*Fres >= fl(i) && (k+0.5)*Fres <= fu(i))
                %Line inside frequency group
                P(n,i) = P(n,i)+Fsp(n,k);
            elseif ((k-0.5)*Fres < fl(i) && (k+0.5)*Fres > fu(i))
                %Frequency group inside
                P(n,i) = P(n,i) + Fsp(n,k)*(fu(i)-Fl(i))/Fres;
            elseif ((k-0.5)*Fres < fl(i) && (k+0.5)*Fres > fl(i))
                %Left Border
                P(n,i) = P(n,i)+Fsp(n,k)*((k+0.5)*Fres-fl(i))/Fres;
            elseif ((k-0.5)*Fres < fu(i) && (k+0.5)*Fres > fu(i))
                %Right Border
                P(n,i) = P(n,i)+Fsp(n,k)*(fu(i)-(k-0.5)*Fres)/Fres;
            else
                %Line outside frequency group
                P(n,i) = P(n,i)+0;
            end
        end
        %Limit result
        P(n,i) = max(P(n,i),0.000000000001);
    end
end

