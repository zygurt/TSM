function [ est_x, est_x_2 ] = LPC_residual( x, fs, N, H )
%[ est_x, est_x_2 ] = LPC_residual( x, fs, N, H )
%   Generate residual signals using Linear Prediction Coefficients
%   WIP

ms = (N/fs)*1000;

x_buf = buffer( x , N , N-H);

%Compute LPCs
lpc_order = ceil(ms/2);
a = lpc(x_buf,lpc_order);
%Produce Residual Signal
est_x = filter(a(1:end),1,x_buf);  % Estimated signal (Residual Signal)
%To inverse the operation
% y = filter(1,a(1:end),est_x);
% r = x_buf-est_x;
% r = r/max(max(r));
% 
% mesh(r)
% view(0,90)




%Compute apks
apk_order = ceil(ms/2);

w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))); %Hann window
W = repmat(w,1,size(x_buf,2));
X_buf = fft(x_buf.*W);

%Compute Smooth Spectrum
X_buf_POWER = abs(X_buf).^2;
power = ifft(X_buf_POWER);

est_x_2 = zeros(size(power));
for m = 1:size(power,2)
    auto_corr = xcorr(power(:,m),apk_order+1);
    apks = levinson(auto_corr,apk_order);
    
    smooth_spectrum = 1./(abs(fft(apks,N)).^2);
    
    est_x_2(:,m) = abs(X_buf(:,m)) ./smooth_spectrum';
end
% mag_est_x_2 = abs(est_x_2(1:end/2+1,:));

disp('LPC done')
end

