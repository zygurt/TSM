function [ p ] = trans_loc( mag, sensitivity, x )
%[ p ] = trans_loc( mag, sensitivity, x )
%   Transient Locations
%   mag: Magnitude spectrum
%   sensitivity: How sensitive to peaks the function is. (Not currently
%   used, as local peak doesn't use it)

% N = size(mag,1);
% W = (1:N).^2;
% W = repmat(W',1,size(mag,2));
% E_tilda_part = W.*mag.^2;
E_tilda_part = mag;

E_tilda = sum(E_tilda_part,1);
% E_tilda(E_tilda==0) = eps(0);
E_tilda_log = log10(E_tilda);
% E_tilda_log(E_tilda_log==-Inf) = eps(0);
E_tilda_log = E_tilda_log(2:end)+E_tilda_log(1:end-1);
E_log_tilda_diff = E_tilda_log(2:end)-E_tilda_log(1:end-1);


E_log_tilda_diff(E_log_tilda_diff==-Inf) = 0;
E_log_tilda_diff(E_log_tilda_diff==Inf) = 0;
E_log_tilda_diff(isnan(E_log_tilda_diff)) = 0;

% figure
% subplot(311)
% plot(x)
% title('Original')
% subplot(312)
% plot(E_tilda_log);
% title('Log10 E tilda');
% subplot(313)
% plot(E_log_tilda_diff);
% title('E log10 tilda diff')

p = local_peak(E_log_tilda_diff, 20, sensitivity); %Resolution of 125 ms
% if(length(p)>1)
%     p = p(1);
% end
% tr_loc = p*N*(1-overlap); %Convert frame to sample location


end

