function [ y ] = SliceTSM_overlap( x, Fs, TSM, s )
%[ y ] = SliceTSM( x, TSM )
%   This TSM method deconstructs signal into slices and realigns on a new
%   grid


x = sum(x,2);
x = x/max(abs(x));


[ p ] = trans_loc( mag, s, x )




%% ------Spectral Features--------

ms = 16;
overlap = 0.75;
N = 2^nextpow2(Fs*ms*10^(-3));
x_buf = buffer(x,N,overlap*N);
X = fft(x_buf,N);

X_mag = abs(X);
W = (1:N).^2;
W = repmat(W',1,size(X_mag,2));
E_tilda_part = W.*X_mag.^2;

E_tilda = sum(E_tilda_part,1);
E_tilda_log = log10(E_tilda);

E_log_tilda_diff = E_tilda_log(2:end)-E_tilda_log(1:end-1);

%Plot the Original and Energy
figure
subplot(311)
plot(x)
title('Original')
subplot(312)
plot(E_tilda_log);
title('Log10 E tilda');
subplot(313)
plot(E_log_tilda_diff);
title('E log10 tilda diff')



%%
%Find peak locations in derivative as transient locations
p = local_peak(E_log_tilda_diff, 20, s); %Resolution of 125 ms
tr_loc = p*N*(1-overlap); %Convert frame to sample location

%Plot the transient locations
figure(1)
subplot(311)
hold on
y_line = ones(size(tr_loc));
line([tr_loc;tr_loc],[-1*y_line ; y_line],'Color','black','LineStyle',':');
subplot(313)
line([p;p],[-1*max(E_log_tilda_diff) ; max(E_log_tilda_diff)],'Color','black','LineStyle',':');
hold off

%Compute segments in the original
seg_start = [1,tr_loc];
seg_end = [tr_loc-1,length(x)];
seg_len = seg_end-seg_start;

TSM_tr = ceil(seg_start/TSM);

y = zeros(ceil(length(x)/TSM),1);
silence_end = 0;
silence_start = TSM_tr+seg_len;
silence_sections = zeros(length(seg_start)-1,2);
N = 2048;
for tr = 1:length(seg_start)
    y(TSM_tr(tr):TSM_tr(tr)+seg_len(tr)) = y(TSM_tr(tr):TSM_tr(tr)+seg_len(tr)) + x(seg_start(tr):seg_end(tr));
%     frame = y(TSM_tr(tr)+seg_len(tr)-N+1:TSM_tr(tr)+seg_len(tr));
%     FRAME = fft(frame,N); %Add windowing
    
%     [env_up,env_low] = envelope(frame,250,'peak');
%     env = (env_up+env_low)/2;
    
%     p = polyfit(1:length(env),env',4);
    
    %Plot the envelope on the signal
    %Plot the continuation of the envelope on the signal
    
    
    
    
    if tr<length(seg_start)
        silence_sections(tr,:) =[TSM_tr(tr)+seg_len(tr) TSM_tr(tr+1)];
    end
end



end

