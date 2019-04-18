function [ y, ZFR_N ] = FESOLA( x, N, TSM, fs )
%[ y ] = ESOLA( x, fs, TSM )
%   x = input signal
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
    disp('This ESOLA method currently only works for mono signals.');
    y = 0;
    return;
end

alpha = 1/TSM;
Ss = N/2;
Sa = round(Ss/alpha);

w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))); %hanning window

y = zeros(ceil(length(x)*alpha),num_chan); %1.1 is to allow for non exct time scaling
y_epochs = zeros(length(y),num_chan);
win = zeros(length(y),num_chan);

[epochs, ZFR_N] = ZFR(x, 1, fs, 1);

%Fuzzy epochs to increase alignment
epochs = [0; epochs; 0];
e = find(epochs);
fuzzy_epoch = epochs;
fuzzy_epoch(e-1) = 0.6;
fuzzy_epoch(e+1) = 0.6;
epochs = fuzzy_epoch(2:end-1);

y(1:N,:) = x(1:N,:);
y_epochs(1:N,:) = epochs(1:N,:);
win(1:N,:) = w(1:N,:);
m = 1;
while m*Sa<length(x)-2*N
    %Create epoch frames
    in_epoch = epochs(m*Sa+1:m*Sa+N);
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
       
        %This priorities the maximum for forwards and backwards
        %Find overall max in each correlation, then find the location
        [max_correlation, loc] = max([k_arr; k_arr2]);
        
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
        return;
    end
    %Calculate the new input grain
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

    %Plotting
%     F=figure(4);%,'Position',[0 0 500 300])
%     F.Position = [1920-500 200 500 300];
%     plot(m*Ss+1:m*Ss+N,out_epoch+1,'k--')
%     hold on
%     plot(m*Ss+1:m*Ss+N,epochs(m*Sa+km+1:m*Sa+N+km),'k');
%     hold off
%     title('Aligned Epochs Using FESOLA');
%     xlabel('Time (Samples)')
%     ylabel('Epoch Amplitude')
%     legend('Output Epochs','Input Epochs','Location', 'SouthEast')
%     axis([m*Ss+1,m*Ss+floor(N/2), -0.1 2.1])
%     
%     
%     if(max(in_epoch)>0)
%         if input('Save frame? ')
%             print('../FESOLA_Epoch_Alignment','-dpng')
%             print('../FESOLA_Epoch_Alignment','-depsc')
%         end
%         disp('movement')
%     end
    
    
    %Increase the frame counter
    m = m+1;
    
end
win(win<0.98) = 1;
y = y./win;

y = y/max(abs(y));
%Truncate to the end of the signal
% y = y(1:m*Ss+N);

end

%OLD Code for plotting testing


%     figure(1)
%     subplot(311)
%     plot(out_epoch)
%     title('Out epoch')
%     subplot(312)
%     plot(in_epoch)
%     title('In epoch')
%     subplot(313)
%     plot(epochs(m*Sa+km+1:m*Sa+km+N));
%     title('Adjusted Epoch');
%     
%     %     figure(2)
%     %     subplot(211)
%     %     plot(x(m*Sa+1:m*Sa+N).*w);
%     %     title('In Grain Initial')
%     %     %Select the final analysis frame
%     %     subplot(212)
%     %     plot(in_grain)
%     %     title('In Grain Final')
%     
%     figure(2)
%     plot(m*Ss+1:m*Ss+N,out_epoch+1)
%     hold on
%     plot(m*Ss+1:m*Ss+N,epochs(m*Sa+km+1:m*Sa+N+km));
%     hold off
%     title('Aligned km epochs');
%     legend('Output','Next Frame');
%     
%     figure(3)
%     subplot(211)
%     plot(k_arr)
%     title('k array')
%     subplot(212)
%     plot(k_arr2)
%     title('k array 2')
%     
%     figure(4)
%     plot(m*Ss+1:m*Ss+N,out_epoch+1)
%     hold on
%     plot(m*Ss+1:m*Ss+N,epochs(m*Sa+k2+1:m*Sa+N+k2));
%     hold off
%     title('Aligned k2 epochs');
%     legend('Output','Next Frame');
%     
%     figure(5)
%     plot(m*Ss+1:m*Ss+N,out_epoch+1)
%     hold on
%     plot(m*Ss+1:m*Ss+N,epochs(m*Sa+k4+1:m*Sa+N+k4));
%     hold off
%     title('Aligned k4 epochs');
%     legend('Output','Next Frame');
%     
%     figure(6)
%     plot(m*Ss+1:m*Ss+N,out_epoch+1)
%     hold on
%     plot(m*Ss+1:m*Ss+N,epochs(m*Sa+k7+1:m*Sa+N+k7));
%     hold off
%     title('Aligned k7 epochs');
%     legend('Output','Next Frame');
    


% figure
% plot(k_array)
% title('km');
% figure
% plot(k2_array)
% title('k2');
% figure
% plot(k3_array)
% title('k3');
% figure
% plot(k4_array)
% title('k4');
%
% figure
% plot(win)
% axis([10000 20000 0.9 1.1])
% figure
% winf = abs(fft(win(10000:20000,:)-1));
% winf = winf/max(winf);
% plot(20*log10(winf(1:round(length(winf)/2))));
