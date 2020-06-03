function [ N, N_total, E_Thresh ] = loudness( E, fc, version, model )
%[ N, N_total ] = loudness( E, fc, version, model )
%   Implemented as per ITU-R BS.1387-1 Section 3.3
global debug_var

if debug_var
disp('  Loudness')
end
if (strcmp(model, 'fft') || strcmp(model, 'FFT'))
    const = 1.07664;
    if(strcmp(version, 'basic') || strcmp(version, 'Basic'))
        Z = 109;
    else
        Z = 55;
    end
elseif (strcmp(model, 'fb') || strcmp(model, 'FB'))
    const = 1.26539;
    Z = 40;
else
    disp('Unknown model')
    N = 0;
    return
end

E_Thresh = 10.^(0.364*(fc/1000).^-0.8);

s = 10.^(0.1*(-2-2.05*atan(fc/4000)-0.75*atan((fc/1600).^2)));

N = zeros(size(E));
N_total = zeros(size(E,1),1);
for n = 1: size(E,1)
    N(n,:) = (const*((1./s).*(E_Thresh/10^4)).^0.23) .* ...
        ((1-s+((s.*E(n,:))/E_Thresh)).^0.23-1);
    N_total(n) = (24/Z)*sum(max(N(n,:),0),2);
end

end

