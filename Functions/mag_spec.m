function [ m, p ] = mag_spec( x, N, H )
%[ m, p ] = mag_spec( x, N, H )
%   This function returns the magnitude(m) and phase(p) spectra of signal x
%   x is the signal
%   N is the frame length
%   H is the hop size
%   Uses Hann window

x_buf = buffer(x,N,N-H);
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))); %Hann window
W = repmat(w,1,size(x_buf,2));
X_buf = fft(x_buf.*W);
m = abs(X_buf(1:N/2+1,:));
p = angle(X_buf(1:N/2+1,:));
end

