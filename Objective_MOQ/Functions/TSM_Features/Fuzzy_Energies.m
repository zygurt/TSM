function [Es,Et,En] = Fuzzy_Energies(x,General)
%[Es,Et,En] = Fuzzy_Energies(x,General)
%   x is the input signal
%   General is a structure containing side information.
%   Implementation follows that of [1], and uses median filtering code of [2].
%References:
%[1] Fierro, L., & Välimäki, V. Towards Objective Evaluation of Audio Time-Scale Modification Methods.
%[2] Damskägg, E. P., & Välimäki, V. (2017). Audio time stretching using fuzzy classification of spectral bins. Applied Sciences, 7(12), 1293.


%Comment the following 2 lines when actually using as they will be in
%General already
% General.fs = 44100;
% General.N = 2048;
N = General.N;
w = PEAQ_Hann( N )';
x_buf = buffer(x,N,512);
x_win = x_buf.*repmat(w,1,size(x_buf,2));
X = fft(x_win,N);

X_mag = abs(X(1:(N/2+1),:));

%---From Fuzzy PV implementation [2]----
filter_length_t = 200e-3; % in ms
filter_length_f = 500; % in Hz
nMedianH = round(filter_length_t * General.fs / 512);
nMedianV = round(filter_length_f* N / General.fs);

%Median filtering
%Vertical
Xt = medfilt1(X_mag,nMedianV,[],1);
%Horizontal
Xs = medfilt1(X_mag,nMedianH,[],2);

Rs = Xs./(Xs+Xt);
Rt = Xt./(Xs+Xt);
Rs(isnan(Rs)) = 0;
Rt(isnan(Rt)) = 0;
%-----------------------------------
Rn = 1-abs(Rs-Rt);


Xs = X_mag.*Rs;
Xt = X_mag.*Rt;
Xn = X_mag.*Rn;

Es = sum(Xs.^2,1);
Et = sum(Xt.^2,1);
En = sum(Xn.^2,1);
end

