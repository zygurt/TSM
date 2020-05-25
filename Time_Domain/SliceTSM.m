function [ y ] = SliceTSM( x, Fs, TSM )
%[ y ] = SliceTSM( x, TSM )
%   This TSM method deconstructs signal into slices and realigns on a new
%   grid

x = sum(x,2);
x = x/max(abs(x));


%% ------Temporal Features--------
% figure
% subplot(511)
% plot(x)
% title('Original')
% %Find the transient locations
% N = 64;
% w = hann(N);
% E_0 = zeros(size(x));
% E = zeros(size(x));
% %Create envelope of the signal
% for n = 1:length(x)-N
%     E_0(n) = sum(abs(x(n:n+N-1)).*w)/N;
% end
% subplot(512)
% plot(E_0)
% title('Amplitude Envelope')
% 
% %Create envelope of the signal
% for n = 1:length(x)-N
%     E(n) = sum(x(n:n+N-1).^2.*w)/N;
% end
% subplot(513)
% plot(E)
% title('Local Energy Envelope')
% 
% %Calculate the first derivative of the signal
% E_0_deriv = E_0(2:end)-E_0(1:end-1);
% 
% subplot(514)
% plot(E_0_deriv)
% title('Derivative of Amplitude Energy')
% 
% %Calculate the first derivative of the signal
% E_0_log_deriv = log(E_0(2:end))-log(E_0(1:end-1));
% 
% subplot(515)
% plot(E_0_log_deriv)
% title('Derivative of Log Local Energy')


%% ------Spectral Features--------

% LinSpectrogram(x,44100,25,1);

ms = 16;
N = 2^nextpow2(Fs*ms*10^(-3));
x_buf = buffer(x,N);%,0.75*N);
X = fft(x_buf,N);

X_mag = abs(X);
W = (1:N).^2;
W = repmat(W',1,size(X_mag,2));
E_tilda_part = W.*X_mag.^2;
% figure
% mesh(E_tilda_part);
% view([0 90])

E_tilda = sum(E_tilda_part,1);
% figure
% subplot(511)
% plot(x)
% title('Original')
% subplot(512)
% plot(E_tilda);
% title('E tilda');
% subplot(513)
% plot(log10(E_tilda));
% title('Log10 E tilda');

% E_tilda_diff = E_tilda(2:end)-E_tilda(1:end-1);

% subplot(514)
% plot(E_tilda_diff)
% title('E tilda diff');

E_log_tilda_diff = log10(E_tilda(2:end))-log10(E_tilda(1:end-1));

% subplot(515)
% plot(E_log_tilda_diff);
% title('E log10 tilda diff')


%%


%Compare peak locations to transient locations

% [peaks] = find_peaks(x_deriv);
% p = find(E_log_tilda_diff>mean(E_log_tilda_diff));%2);
p = find(E_log_tilda_diff>mean(E_log_tilda_diff)+std(E_log_tilda_diff));%2);
tr_loc = p*N;
%Plot the Transient locations
% subplot(511)
% hold on
% y_line = ones(size(tr_loc));
% line([tr_loc;tr_loc],[-1*y_line ; y_line],'Color','black');
% hold off

seg_start = [1,tr_loc];
seg_end = [tr_loc-1,length(x)];
seg_len = seg_end-seg_start;

TSM_tr = ceil(seg_start/TSM);

y = zeros(ceil(length(x)/TSM),1);

for tr = 1:length(seg_start)
    y(TSM_tr(tr):TSM_tr(tr)+seg_len(tr)) = y(TSM_tr(tr):TSM_tr(tr)+seg_len(tr)) + x(seg_start(tr):seg_end(tr));
    
end



end

