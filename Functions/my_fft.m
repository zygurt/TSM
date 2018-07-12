function [ X ] = my_fft( x, N )
%[ X ] = my_fft( x, N )
%   My own implementation of the fft
%   Using radix 2 for simplicity
%   Working from https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

if(size(x,1)<size(x,2))
    x = x';
end

if(N<2)
    %Bottom of recursion
    X = x;
else
    x = [x(1:2:end); x(2:2:end)];
    %In a 0 index language, lower half is even.  In a 1 index, the lower
    %half shows odd, but is actually even.
    e = my_fft(x(1:end/2),length(x)/2);
    o = my_fft(x(end/2+1:end),length(x)/2);
    %Recombine
    w = exp((-1i*2*pi*(0:(length(x)/2)-1)')/N);
    X = [e+w.*o ; e-w.*o];
end

end