function [ y ] = ESOLA( x, N, TSM, fs )
%[ y ] = ESOLA( x, fs, TSM )
%   x = input signal
%   N = frame size (Normally 3 to 4 pitch periods) (20ms used in paper)
%   TSM = Time scale
%   fs = sampling frequency of signal
%   y = output signal
%   Implementation based on Rudresh et al., Epoch-Synchronous Overlap-Add
%   (ESOLA) for Time- and Pitch-Scale Modification of Speech Signals, 2018

% Tim Roberts - Griffith University 2018

num_chan = size(x,2);
if(num_chan > 1)
    disp('This ESOLA method currently only works for mono signals.');
    y = 0;
    return;
end
%Calculate alpha
a = 1/TSM;
%Calculate Synthesis Shift
Ss = N/2;
%Calculate Analysis Shift
Sa = round(Ss/a);
%Generate Hanning window
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
%Initialise output
y = zeros(ceil(length(x)*a),num_chan); %1.1 is to allow for non exct time scaling
%Initialise output epochs
y_epochs = zeros(length(y),num_chan);
%Initialise window overlap output
win = zeros(length(y),num_chan);
%Generate epochs
[epochs, ~] = ZFR(x, 1, fs, 2);
%Copy frame 0
y(1:N,:) = x(1:N,:);
y_epochs(1:N,:) = epochs(1:N,:);
win(1:N,:) = w(1:N,:);
%Increment frame
m = 1;
while m*Sa<length(x)-2*N
    %Create epoch frames
    in_epoch = epochs(m*Sa+1:m*Sa+N);
    out_epoch = y_epochs(m*Ss+1:m*Ss+N);
    %Calculate delay for epoch match
    if(max(in_epoch) && max(out_epoch))
        %Epoch(s) in the current frames
        %Find epochs in output frame
        lm = find(out_epoch==1);
        %Find epochs in input frame
        nm = find(in_epoch==1);
        %Find distance from first output epoch to all input epochs.
        k = nm-lm(1);
        %Find the number of negative values to exclude them from the search
        k_diff = find(k<0);
        if length(k_diff)==length(k)
            km=0;
        else
        %km is the smallest value >= 0
            km = min(k(length(k_diff)+1:end));
        end
    else
        %No epochs in one or either frame
        km = 0;
    end
    %Double check accessing bounds
    if(m*Sa+N+km)>length(x)
        disp('Accessed beyond original signal.  Returning signal processed thus far.')
        return;
    end
    %Extract grain to overlap
    in_grain = x(m*Sa+km+1:m*Sa+N+km).*w;
    %Overlap and add the new frame
    y(m*Ss+1:m*Ss+N,:) = y(m*Ss+1:m*Ss+N,:)+in_grain;
    %Overlap and add the epochs for the new frame
    y_epochs(m*Ss+1:m*Ss+N,:) = y_epochs(m*Ss+1:m*Ss+N,:)+epochs(m*Sa+km+1:m*Sa+N+km);
    %Overlap the window
    win(m*Ss+1:m*Ss+N,:) = win(m*Ss+1:m*Ss+N,:)+w(1:N,:);
    
    %Lets do some plotting
%     F=figure(4);%,'Position',[0 0 500 300])
%     F.Position = [1920-500 200 500 300];
%     plot(m*Ss+1:m*Ss+N,out_epoch+1,'k--')
%     hold on
%     plot(m*Ss+1:m*Ss+N,epochs(m*Sa+km+1:m*Sa+N+km),'k');
%     hold off
%     title('Aligned Epochs Using ESOLA');
%     xlabel('Time (Samples)')
%     ylabel('Epoch Amplitude')
%     legend('Output Epochs','Input Epochs','Location', 'SouthEast')
%     axis([m*Ss+1,m*Ss+floor(N/2), -0.1 2.1])
%     
%     
%     if(max(in_epoch)>0)
%         if input('Save frame? ')
%             print('../ESOLA_Epoch_Alignment','-dpng')
%             print('../ESOLA_Epoch_Alignment','-depsc')
%         end
%         disp('movement')
%     end
    
    
    %Increase the frame counter
    m = m+1;
end
%Normalise to window overlap
win(win<0.98) = 1;
y = y./win;
%Normalise overall
y = y/max(abs(y));
end
