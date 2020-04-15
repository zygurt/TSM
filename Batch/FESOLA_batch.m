function [ y ] = FESOLA_batch( x, N, TSM, fs, filename )
%[ y ] = FESOLA_batch( x, N, TSM, fs, filename )
%   x = input signal1
%   N = frame size (Normally 3 to 4 pitch periods) (20ms used in paper)
%   TSM = Time scale
%   fs = sampling frequency of signal
%   y = output signal
%   Initial Implementation based on Rudresh et al., Epoch-Synchronous Overlap-Add
%   (ESOLA) for Time- and Pitch-Scale Modification of Speech Signals, 2018
%   Improvements by Tim Roberts - Griffith University 2018

% Tim Roberts - Griffith University 2018

num_chan = size(x,2);
if(num_chan > 1)
    disp('This FESOLA method currently only works for mono signals.');
    disp('Summing to mono.')
    x = sum(x,2);
end
num_chan = size(x,2);
x = x/max(abs(x));
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))); %hanning window

[epochs, ~] = ZFR(x, 1, fs, 1);

% figure
% plot((epochs*2)-1)
% hold on
% plot(x/max(abs(x)))
% hold off


%Fuzzy epochs to increase alignment
epochs = [0; epochs; 0];
e = find(epochs);
fuzzy_epoch = epochs;
fuzzy_epoch(e-1) = 0.6;
fuzzy_epoch(e+1) = 0.6;
epochs = fuzzy_epoch(2:end-1);

for t = 1:length(TSM)
    fprintf('%s, FESOLA, %g%%\n',filename, TSM(t)*100);
    tsm = TSM(t);

    alpha = 1/tsm;
    Ss = N/2;
    Sa = round(Ss/alpha);

    y = zeros(ceil(length(x)*alpha),num_chan); %1.1 is to allow for non exct time scaling
    y_epochs = zeros(length(y),num_chan);
    win = zeros(length(y),num_chan);

    y(1:N,:) = x(1:N,:);
    y_epochs(1:N,:) = epochs(1:N,:);
    win(1:N,:) = w(1:N,:);
    m = 1;

    while m*Sa<length(x)-2*N
        %Create epoch frames
        in_epoch = epochs(m*Sa+1:m*Sa+N);

        if(m*Ss+N>size(y_epochs,1))
          %Sometimes the output vector isn't long enough for the final frame
          %This concatenates silence to the end of the output signals.
          y = [y ; zeros((m*Ss+N)-size(y,1),1)];
          y_epochs = [y_epochs ;zeros((m*Ss+N)-size(y_epochs,1),1)];
          win = [win ; zeros((m*Ss+N)-size(win,1),1)];
        end

        out_epoch = y_epochs(m*Ss+1:m*Ss+N);

        if(max(in_epoch) && max(out_epoch))
            %Generate cross correlations for epochs
            k_arr = zeros(round(0.75*Ss),1);
            k_arr2 = zeros(round(0.75*Ss),1);
            for k = 1:round(0.75*Ss)
                k_arr(k) = sum(out_epoch(1:end-k+1).*in_epoch(k:end));
                k_arr2(k) = sum(out_epoch(k:end).*in_epoch(1:end-k+1));
            end
            %This could be optimised for speed by removing fuzzyness and
            %summing after element-wise AND operations

            %Find overall max in each correlation, then find the location
            [max_correlation, loc] = max([k_arr k_arr2]);

            if loc < length(k_arr)
                km = min(find(k_arr==max_correlation))-1;
            else
                km = -1*(min(find(k_arr2==max_correlation))-1);
            end
            if(isempty(km))
                km = 0;
            end
        else
            km = 0;
        end

        %Double check accessing bounds
        if(m*Sa+N+km)>length(x)
            disp('Accessed beyond original signal.  Returning signal processed thus far.')
            break;
        end
        %Extract the aligned input grain
        if(m*Sa+km+1>0)
            in_grain = x(m*Sa+km+1:m*Sa+N+km).*w;
        else
            in_grain = x(m*Sa+1:m*Sa+N).*w;
        end

        %Overlap and add the new frame
        y(m*Ss+1:m*Ss+N,:) = y(m*Ss+1:m*Ss+N,:)+in_grain;
        %Overlap and add the epochs for the new frame
        y_epochs(m*Ss+1:m*Ss+N,:) = y_epochs(m*Ss+1:m*Ss+N,:)+epochs(m*Sa+km+1:m*Sa+N+km);
        %Overlap the window
        win(m*Ss+1:m*Ss+N,:) = win(m*Ss+1:m*Ss+N,:)+w(1:N,:);

        %Increase the frame counter
        m = m+1;
        %     figure(2)
        %     plot(y);


    end
    win(win<0.98) = 1;
    y = y./win;

    y = y/max(abs(y));

    out_filename = sprintf('%s_%g_per.wav',filename,TSM(t)*100);
    audiowrite(out_filename,y,fs);
end
end
