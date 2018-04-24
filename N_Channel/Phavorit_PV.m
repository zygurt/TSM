function [ y_out ] = Phavorit_PV( x, N, TSM, PL )

%This Phase Locking Phase Vocoder processes each channel individually.
%   x is the input signal
%   N is the frame length.  Must be power of 2.
%   TSM is the TSM ratio 0.5 = 50%, 1 = 100% and 2.0 = 200% speed
%   PL = 0 for identity, 1 for scaled

% Tim Roberts - Griffith University 2018

%Calculate the number of channels
num_chan = size(x,2);
%Set time scale parameters
alpha = 1/TSM;  % alpha<1 = compression, alpha>1 = expansion
beta = (2+alpha)/3;  % beta = 1 should be identity method, beta=alpha is another option
%beta = alpha; %suggestion from Laroche and Dolson
%Set Hop Factors
Hs = N/4;
Ha = Hs/alpha;
%Create window function (Hann) for correct number of channels
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
wn = repmat(w,1,num_chan);
%Zero pad and normalise the input
x = [zeros(N,num_chan); x; zeros(N-ceil(mod(length(x),Ha)),num_chan)]/max(max(abs(x)));
%Initialise the output
y = zeros(2*N+ceil(length(x)*alpha),num_chan);
%Set omega for looped calculations
omega_k = repmat(2*pi*(0:N/2)'/N,1,num_chan);
%Set the initial phases  (Only half the fft is needed)
prev_X_phase = zeros (N/2+1,num_chan) ;
prev_Y_phase = zeros (N/2+1,num_chan) ;
%Initialise arrays for peak locations
for c = 1:num_chan
    pp(c).a = []; %previous peak array
    pp(c).rl = []; %previous peak region lower bound
    pp(c).ru = []; %previous peak region upper bound
end
%using pointers set the analysis frame
ptr_input = 1;
ptr_output = 1;
ptr_end = length(x)-N; %When ptr_in = ptr_end, stop
frame = 1;

while(ptr_input<ptr_end)
    frame_current = x(ptr_input:ptr_input+N-1,:).*wn;
    FRAME_CURRENT = fft(frame_current);
    FRAME_CROP = FRAME_CURRENT(1:N/2+1,:); %Take only what's needed
    %Calculate Magnitude and phase
    mag = abs(FRAME_CROP);
    frame_phase = angle(FRAME_CROP);
    
    switch PL
        case 0
            %Code for Identity Phase Locking (Laroche and Dolson 1999)
            %Find the magnitude spectrum peaks
            [p] = find_peaks_log(mag);  %p=> peaks struct
            for c = 1:num_chan
                if ~ p(c).empty_flag
                    %At each peak calc inst_freq and synthesis phase eq(6)
                    for n = 1:length(p(c).pa)
                        delta_phi(n) = frame_phase(p(c).pa(n))-prev_X_phase(p(c).pa(n))-Ha*omega_k(p(c).pa(n));
                    end
                    k = round(delta_phi/(2*pi)) ;
                    delta_phi_adjust = delta_phi-k*2*pi;
                    for n= 1:length(p(c).pa)
                        inst_freq(n) = omega_k(p(c).pa(n)) + delta_phi_adjust(n)/Ha; %Element 1 is peak 1
                        Y_phase(n) = prev_Y_phase(p(c).pa(n)) + Hs*inst_freq(n);
                    end
                    %Calc theta and phasor eq(13)
                    theta = zeros((N/2)+1,1);
                    for n = 1:length(p(c).pa)
                        theta(p(c).rl(n):p(c).ru(n)) = Y_phase(n) - frame_phase(p(c).pa(n));
                    end
                    Z = exp(1i*theta);
                    %Apply rotation to all channels in region eq(14)
                    Y(:,c) = Z.*FRAME_CROP(:,c);
                else
                    Y(:,c) = FRAME_CROP(:,c);
                end
            end
            prev_X_phase = frame_phase;
            prev_Y_phase = angle(Y);
            %Convert Y back to full length FFT
            %Create y_frame
            y_frame = real(ifft([Y;conj(Y(end-1:-1:2,:))])).*wn;
            
        case 1
            %Code for Scaled Phase Locking
            %Find peaks in current frame
            [p] = find_peaks_log(mag); %p=> peaks struct
            for c = 1:num_chan
                if ~ p(c).empty_flag
                    for n = 1:length(p(c).pa)
                        %For each peak in this frame, find corresponding peak in
                        %previous frame
                        prev_peak_k = previous_peak_heuristic(p(c).pa(n), pp(c).a, pp(c).rl, pp(c).ru);
                        if prev_peak_k > 0
                            %Calc inst_freq and synthesis phase eq(15)
                            %UP TO HERE
                            delta_phi_s(n) = frame_phase(p(c).pa(n),c)-prev_X_phase(prev_peak_k)-Ha*omega_k(p(c).pa(n),c);
                            k_s = round(delta_phi_s(n)/(2*pi)) ;
                            delta_phi_adjust_s(n) = delta_phi_s(n)-k_s*2*pi;
                            inst_freq_s(n) = omega_k(p(c).pa(n)) + delta_phi_adjust_s(n)/Ha;
                            Y_synth_phase_s(n) = prev_Y_phase(prev_peak_k) + Hs*inst_freq_s(n);
                            %Calc analysis difference between peak channel and current channel eq(16)
                            Y_phase(p(c).rl(n):p(c).ru(n),c) = repelem(Y_synth_phase_s(n),p(c).ru(n)-p(c).rl(n)+1) + ...
                                beta*(frame_phase(p(c).rl(n):p(c).ru(n),c)'- ...
                                repelem(frame_phase(p(c).pa(n),c), p(c).ru(n)-p(c).rl(n)+1));
                        else
                            %Do iPL method
                            delta_phi(p(c).rl(n):p(c).ru(n),c) = frame_phase(p(c).rl(n):p(c).ru(n),c) - prev_X_phase(p(c).rl(n):p(c).ru(n),c)-Ha*omega_k(p(c).rl(n):p(c).ru(n),c);
                            k = round(delta_phi(p(c).rl(n):p(c).ru(n),c)/(2*pi)) ;
                            delta_phi_adjust(p(c).rl(n):p(c).ru(n),c) = delta_phi(p(c).rl(n):p(c).ru(n),c)-k*2*pi;
                            inst_freq(p(c).rl(n):p(c).ru(n),c) = omega_k(p(c).rl(n):p(c).ru(n),c) + delta_phi_adjust(p(c).rl(n):p(c).ru(n),c)/Ha;
                            Y_phase(p(c).rl(n):p(c).ru(n),c) = prev_Y_phase(p(c).rl(n):p(c).ru(n),c) + Hs*inst_freq(p(c).rl(n):p(c).ru(n),c);
                        end
                    end
                    Y(:,c) = mag(:,c).*exp(1i*Y_phase(:,c));
                    pp(c).a = p(c).pa;
                    pp(c).rl = p(c).rl;
                    pp(c).ru = p(c).ru;
                    prev_X_phase = frame_phase;
                    prev_Y_phase(:,c) = angle(Y(:,c));
                    
                else
                    Y = FRAME_CROP;
                    pp(c).a = p(c).pa;
                    pp(c).rl = p(c).rl;
                    pp(c).ru = p(c).ru;
                    prev_X_phase = frame_phase;
                    prev_Y_phase = angle(Y);
                end
            end
            %Convert Y back to full length FFT
            %Create y_frame
            y_frame = real(ifft([Y;conj(Y(end-1:-1:2,:))])).*wn;
            
        otherwise
            error('Incorrect phase locking option')
    end
    %iFFT, circular shift, window
    %y_frame = circshift(real(ifft(Y)),0).*wn;   %length(frame_current)/2
    %Overlap-add
    y(ptr_output+1:ptr_output+N,:) = y(ptr_output+1:ptr_output+N,:) + y_frame;
    %Increase the pointer location
    frame = frame+1;
    ptr_input = round((frame-1)*Ha)+1;
    ptr_output = ptr_output+Hs;
end
%toc
%Normalise the output
y_out = y(N+1:end,:)/max(max(abs(y)));
end
