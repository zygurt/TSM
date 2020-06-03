function [ E2, E ] = fb_tds_calc( E0, General )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

i_vec = 0:11;
cos_vec = cos(pi*((i_vec-5)/12)).^2;
for k = 1:size(E0,1)
    for n = 6:(size(E0,2)-6)/6
        E1(k,n-5) = 0.9761/6*sum(E0(k,6*n-i_vec).*cos_vec);
    end
end

%Adding internal noise
const = fb_constants();
fc = const.fc;

E_thresh = 10.^(0.4*0.364*(fc/1000).^-0.8);
E2 = E1+repmat(E_thresh',[1,size(E1,2)]);

tau_min = 0.004;
tau_100 = 0.02;

tau = tau_min+(100./fc)*(tau_100-tau_min);

a = exp(-192./(General.fs*tau'));

%Define the size of E.
E = zeros(size(E2));
E(:,1) = E2(:,1);
for n = 2:size(E2,2)
    E(:,n) = a.*E(:,n-1)+(1-a).*E2(:,n);
end


end

