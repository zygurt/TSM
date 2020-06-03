function [ xHarm, xPerc ] = HP_seperation( x )
%[ xHarm, xPerc ] = HP_seperation( x )
%   Harmonic Percussive Separation
%   Abstracted from Driedger's hpSep.m in the TSM Toolbox
% x                 input signal.
% xHarm             the harmonic component of the input signal x.
% xPerc             the percussive component of the input signal x.
%

global debug_var

N = 1024;
anaHop = N/4;
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
filLenHarm = 10;
filLenPerc = 10;
maskingMode = 'binary';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% harmonic-percussive separation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% stft
xC_buf = buffer(x,N,N-anaHop);
XC = fft(xC_buf);
magSpec = abs(XC(1:N/2+1,:));

% harmonic-percussive separation
magSpecHarm = medianFilter(magSpec,filLenHarm,2);
magSpecPerc = medianFilter(magSpec,filLenPerc,1);

switch maskingMode
    case 'binary'
        maskHarm = magSpecHarm >  magSpecPerc;
        maskPerc = magSpecHarm <= magSpecPerc;
        
    case 'relative'
        maskHarm = magSpecHarm ./ (magSpecHarm + magSpecPerc);
        maskPerc = magSpecPerc ./ (magSpecHarm + magSpecPerc);
        
    otherwise
        error('maskingMode must either be "binary" or "relative"');
end

specHarm = [maskHarm;maskHarm(end-1:-1:2,:)] .* XC;
specPerc = [maskPerc;maskPerc(end-1:-1:2,:)] .* XC;

% istft
xHarmC = zeros(N+(size(specHarm,2)-1)*anaHop,1);
xPercC = zeros(N+(size(specPerc,2)-1)*anaHop,1);
% figure
for n = 1:size(specHarm,2)-1
    sframe_Harm = real(ifft(specHarm(:,n))).*w;
    sframe_Perc = real(ifft(specPerc(:,n))).*w;
    if (n-1)*anaHop+N > length(xHarmC)
        disp('Stop')
    end
    xHarmC((n-1)*anaHop+1:(n-1)*anaHop+N) = xHarmC((n-1)*anaHop+1:(n-1)*anaHop+N) + sframe_Harm;
    xPercC((n-1)*anaHop+1:(n-1)*anaHop+N) = xPercC((n-1)*anaHop+1:(n-1)*anaHop+N) + sframe_Perc;
    %     subplot(211)
    %     plot(xHarmC)
    %     axis([1, N+(n-1)*anaHop, 1.1*min(xHarmC), 1.1*max(xHarmC)]);
    %     subplot(212)
    %     plot(xPercC)
    %     axis([1, N+(n-1)*anaHop, 1.1*min(xPercC), 1.1*max(xPercC)]);
end

xHarm = xHarmC;
xPerc = xPercC;
if debug_var
%     figure
%     subplot(311)
%     plot(x)
%     title('Original')
%     axis([1, N+(n-1)*anaHop, 1.1*min(x), 1.1*max(x)]);
%     subplot(312)
%     plot(xHarmC)
%     title('Harmonic')
%     axis([1, N+(n-1)*anaHop, 1.1*min(xHarmC), 1.1*max(xHarmC)]);
%     subplot(313)
%     plot(xPercC)
%     title('Percussive')
%     axis([1, N+(n-1)*anaHop, 1.1*min(xPercC), 1.1*max(xPercC)]);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% median filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Y = medianFilter(X,len,dim)

s = size(X);
Y = zeros(s);

switch dim
    case 1
        XPadded = [zeros(floor(len/2),s(2));X;zeros(ceil(len/2),s(2))];
        for i = 1 : s(1)
            Y(i,:) = median(XPadded(i:i+len-1,:),1);
        end
        
    case 2
        XPadded = [zeros(s(1),floor(len/2)) X zeros(s(1),ceil(len/2))];
        for i = 1 : s(2)
            Y(:,i) = median(XPadded(:,i:i+len-1),2);
        end
        
    otherwise
        error('unvalid dim.')
end
end

