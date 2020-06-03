function [ MusNoise ] = Spectral_Kurtosis( x, y, General, match_method )
%[ MusNoise ] = Spectral_Kurtosis( x, y, General, match_method )
%   Spectral Kurtosis is a measure of musical noise implemented based on
%   Torcoli, M., "An Improved Measure of Musical Noise Based on Spectral
%   Kurtosis", 2019 IEEE Workshop on Applications of Signal Processing to
%   Audio and Acoustics, New Paltz, NY, 2019.
%
%   Match options 'up','down','framing','to_ref','to_test'
global debug_var

if debug_var
    disp('Spectral Kurtosis Calculation')
end

%Create Spectra

N = 1024;
w = (0.5*sqrt(8/3)*(1-cos(2*pi*(1:N)/(N-1))))'; %Normally in PEAQ_Hann.m
% w = PEAQ_Hann(N);

%The silence at the start and end of the signals have already been
%truncated previously within the feature calculation.
%This doesn't matter if the files are the same length.
switch match_method
    case 'up'
        x_buf = buffer(x, N, N/2);
        y_buf = buffer(y, N, N/2);
        
        X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        X_MAG = abs(X(1:((size(X,1)/2)+1),:));
        Y = fft(y_buf.*repmat(w,1,size(y_buf,2)));
        Y_MAG = abs(Y(1:((size(Y,1)/2)+1),:));
        if size(X_MAG,2)<size(Y_MAG,2)
            X_MAG_interp = zeros(size(Y_MAG));
            for n = 1:size(Y_MAG,1)
                X_MAG_interp(n,:) = interp1(linspace(0,1,size(X_MAG,2)),X_MAG(n,:),linspace(0,1,size(Y_MAG,2)));
            end
            %Convert spectral power values to dBA
            [ X_dBA ] = dBA_Torcolli( X_MAG_interp, General.fs );
            [ Y_dBA ] = dBA_Torcolli( Y_MAG, General.fs );
            
        else
            Y_MAG_interp = zeros(size(X_MAG));
            for n = 1:size(Y_MAG,1)
                Y_MAG_interp(n,:) = interp1(linspace(0,1,size(Y_MAG,2)),Y_MAG(n,:),linspace(0,1,size(X_MAG,2)));
            end
            %Convert spectral power values to dBA
            [ X_dBA ] = dBA_Torcolli( X_MAG, General.fs );
            [ Y_dBA ] = dBA_Torcolli( Y_MAG_interp, General.fs );

        end
        
    case 'down'
        x_buf = buffer(x, N, N/2);
        y_buf = buffer(y, N, N/2);
        
        X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        X_MAG = abs(X(1:((size(X,1)/2)+1),:));
        Y = fft(y_buf.*repmat(w,1,size(y_buf,2)));
        Y_MAG = abs(Y(1:((size(Y,1)/2)+1),:));
        if size(Y_MAG,2)<size(X_MAG,2)
            X_MAG_interp = zeros(size(Y_MAG));
            for n = 1:size(Y_MAG,1)
                X_MAG_interp(n,:) = interp1(linspace(0,1,size(X_MAG,2)),X_MAG(n,:),linspace(0,1,size(Y_MAG,2)));
            end
            %Convert spectral power values to dBA
            [ X_dBA ] = dBA_Torcolli( X_MAG_interp, General.fs );
            [ Y_dBA ] = dBA_Torcolli( Y_MAG, General.fs );
        else
            Y_MAG_interp = zeros(size(X_MAG));
            for n = 1:size(Y_MAG,1)
                Y_MAG_interp(n,:) = interp1(linspace(0,1,size(Y_MAG,2)),Y_MAG(n,:),linspace(0,1,size(X_MAG,2)));
            end
            %Convert spectral power values to dBA
            [ X_dBA ] = dBA_Torcolli( X_MAG, General.fs );
            [ Y_dBA ] = dBA_Torcolli( Y_MAG_interp, General.fs );
        end
    case 'framing'
        sk_frames = 1:N/2:length(x)-N;
        
        x_buf = vec_buffer(x, N, sk_frames);
        
        frame_loc = [1 floor(sk_frames(2:end)/General.TSM)];
        y_buf = vec_buffer(y, N, frame_loc);
        
        X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        X_MAG = abs(X(1:((size(X,1)/2)+1),:));
        Y = fft(y_buf.*repmat(w,1,size(y_buf,2)));
        Y_MAG = abs(Y(1:((size(Y,1)/2)+1),:));
        
        %Convert spectral power values to dBA
        [ X_dBA ] = dBA_Torcolli( X_MAG, General.fs );
        [ Y_dBA ] = dBA_Torcolli( Y_MAG, General.fs );
        
%         X_dBA_new = X_dBA(2:end,~isinf(sum(Y_dBA(2:end,:),1)));
%         Y_dBA_new = Y_dBA(2:end,~isinf(sum(Y_dBA(2:end,:),1)));
                
        if(size(X_dBA,2)~=size(Y_dBA,2))
            fprintf('%s, ',General.Testname)
            fprintf('Ref has %d frames, Test has %d frames\n',size(X_dBA,2),size(Y_dBA,2));
        end
        
    case 'to_ref'
        x_buf = buffer(x, N, N/2);
        y_buf = buffer(y, N, N/2);
        
        X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        X_MAG = abs(X(1:((size(X,1)/2)+1),:));
        Y = fft(y_buf.*repmat(w,1,size(y_buf,2)));
        Y_MAG = abs(Y(1:((size(Y,1)/2)+1),:));
                
        Y_MAG_interp = zeros(size(X_MAG));
        for n = 1:size(Y_MAG,1)
            Y_MAG_interp(n,:) = interp1(linspace(0,1,size(Y_MAG,2)),Y_MAG(n,:),linspace(0,1,size(X_MAG,2)));
        end
        %Convert spectral power values to dBA
        [ X_dBA ] = dBA_Torcolli( X_MAG, General.fs );
        [ Y_dBA ] = dBA_Torcolli( Y_MAG_interp, General.fs );
        
        
    case 'to_test'
        x_buf = buffer(x, N, N/2);
        y_buf = buffer(y, N, N/2);
        
        X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        X_MAG = abs(X(1:((size(X,1)/2)+1),:));
        Y = fft(y_buf.*repmat(w,1,size(y_buf,2)));
        Y_MAG = abs(Y(1:((size(Y,1)/2)+1),:));
        
        X_MAG_interp = zeros(size(Y_MAG));
        for n = 1:size(Y_MAG,1)
            X_MAG_interp(n,:) = interp1(linspace(0,1,size(X_MAG,2)),X_MAG(n,:),linspace(0,1,size(Y_MAG,2)));
        end
        %Convert spectral power values to dBA
        [ X_dBA ] = dBA_Torcolli( X_MAG_interp, General.fs );
        [ Y_dBA ] = dBA_Torcolli( Y_MAG, General.fs );

        
end

%Remove zero frames (dB = -Inf) and DC bin
X_dBA_new = X_dBA(2:end,~isinf(sum(Y_dBA(2:end,:),1)));
Y_dBA_new = Y_dBA(2:end,~isinf(sum(Y_dBA(2:end,:),1)));


%Limit and Shift to be non-negative (plus)
% thr_old = sqrt(mean(mean(X_dBA_new.^2)))-20;
thr = pow2db(mean(mean(db2pow(X_dBA_new))))-20;
X_plus = max(X_dBA_new,thr)-thr;
% thr = sqrt(mean(mean(Y_dBA_new.^2)))-20;
thr = pow2db(mean(mean(db2pow(Y_dBA_new))))-20;
Y_plus = max(Y_dBA_new,thr)-thr;


% %Discard frames for which N_out = 0 for all bins
X_plus_small = X_plus(:,sum(Y_plus,1)~=0);
Y_plus_small = Y_plus(:,sum(Y_plus,1)~=0);

%Calculate frequency for each bin
f = (1:(size(X_plus_small,1)-1))/(size(X_plus_small,1)-1)*General.fs/2;

Lower_band = [50,750]; %Hz
Middle_band = [750, 6000]; %Hz
Upper_band = [6000 16000]; %Hz

%Calculate bins closest bins to band frequencies
[~, Lower_band_bin(1)] = min(abs(f-Lower_band(1)));
[~, Lower_band_bin(2)] = min(abs(f-Lower_band(2)));
[~, Middle_band_bin(1)] = min(abs(f-Middle_band(1)));
Middle_band_bin(1) = Middle_band_bin(1)+1; %Plus 1 to remove overlapping regions
[~, Middle_band_bin(2)] = min(abs(f-Middle_band(2)));
[~, Upper_band_bin(1)] = min(abs(f-Upper_band(1)));
Upper_band_bin(1) = Upper_band_bin(1)+1; %Plus 1 to remove overlapping regions
[~, Upper_band_bin(2)] = min(abs(f-Upper_band(2)));

%Split into sub-bands
lower_X = X_plus_small(Lower_band_bin(1):Lower_band_bin(2),:);
middle_X = X_plus_small(Middle_band_bin(1):Middle_band_bin(2),:);
upper_X = X_plus_small(Upper_band_bin(1):Upper_band_bin(2),:);
lower_Y = Y_plus_small(Lower_band_bin(1):Lower_band_bin(2),:);
middle_Y = Y_plus_small(Middle_band_bin(1):Middle_band_bin(2),:);
upper_Y = Y_plus_small(Upper_band_bin(1):Upper_band_bin(2),:);

%Use equation 3 to compute kurt_Nin and kurt_Nout
kurt_lower_X = Kurtosis(lower_X);
kurt_middle_X = Kurtosis(middle_X);
kurt_upper_X = Kurtosis(upper_X);
kurt_lower_Y = Kurtosis(lower_Y);
kurt_middle_Y = Kurtosis(middle_Y);
kurt_upper_Y = Kurtosis(upper_Y);

%compute Sub-band absolute instantaneous log-kurtosis ratio
delta_kurt_lower = abs(log(kurt_lower_Y./kurt_lower_X));
delta_kurt_middle = abs(log(kurt_middle_Y./kurt_middle_X));
delta_kurt_upper = abs(log(kurt_upper_Y./kurt_upper_X));

%Limit values to 0.5
delta_kurt_lower(delta_kurt_lower>0.5) = 0.5;
delta_kurt_middle(delta_kurt_middle>0.5) = 0.5;
delta_kurt_upper(delta_kurt_upper>0.5) = 0.5;

%Remove NaN values

delta_kurt_lower(isnan(delta_kurt_lower)) = 0;
delta_kurt_middle(isnan(delta_kurt_middle)) = 0;
delta_kurt_upper(isnan(delta_kurt_upper)) = 0;


%Testing code
% delta_kurt_lower(delta_kurt_lower>1) = 1;
% delta_kurt_middle(delta_kurt_middle>1) = 1;
% delta_kurt_upper(delta_kurt_upper>1) = 1;

%calculate sub-band energy-weights
w_lower = 10*log10(1/size(lower_Y,1)*sum(10.^(lower_Y/10)));
w_middle = 10*log10(1/size(middle_Y,1)*sum(10.^(middle_Y/10)));
w_upper = 10*log10(1/size(upper_Y,1)*sum(10.^(upper_Y/10)));

%Compute energy weighted mean of instantaneous log-kurtosis ratio
%Use the band which has the maximum value

W_lower = sum(w_lower);
W_middle = sum(w_middle);
W_upper = sum(w_upper);

temp_lower = sum(w_lower.*delta_kurt_lower)/W_lower;
temp_middle = sum(w_middle.*delta_kurt_middle)/W_middle;
temp_upper = sum(w_upper.*delta_kurt_upper)/W_upper;

MusNoise = [temp_lower temp_middle temp_upper max([temp_lower temp_middle temp_upper],[],'omitnan')];%delta_kurt_PI;
MusNoise(isnan(MusNoise)) = 0;

end

