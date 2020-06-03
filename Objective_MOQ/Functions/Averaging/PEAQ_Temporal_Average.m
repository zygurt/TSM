function [ AvgX, RmsX, WinX ] = PEAQ_Temporal_Average( X, method, Z, W)
%[ AvgX, RmsX, WinX ] = PEAQ_Temporal_Averaging( X, method, Z, W)
%   As described by ITU-R BS.1387-1 Section 5.2.1 to 5.2.3

if nargin<=2
    Z = 0;
end

if nargin<=3
    switch(method)
        case 'Linear'
            if(isempty(X))
                AvgX = 0;
                RmsX = 0;
                WinX = 0;
            else
                AvgX = mean(X);
                RmsX = 0;
                WinX = 0;
            end
        case 'Squared'
            RmsX = rms(X);
            AvgX = 0;
            WinX = 0;
        case 'Windowed'
            if(Z>40)
                L = 4;
            else
                L = 25;
            end
            N = length(X);
            temp_n = 0;
            for n = L:N
                temp_i = 0;
                for i = 0:L-1
                    temp_i = temp_i+sqrt(X(n-i));
                end
                temp_i = (temp_i/L).^4;
                temp_n = temp_n + temp_i;
            end
            WinX = sqrt(1/(N-L+1)*temp_n);
            AvgX = 0;
            RmsX = 0;
        otherwise
            AvgX = 0;
            RmsX = 0;
            WinX = 0;
    end
    
else
    switch(method)
        case 'Linear'
            AvgX = sum(W.*X)/sum(W);
            RmsX = 0;
            WinX = 0;
        case 'Squared'
%             RmsX = sqrt(Z)*sqrt(((W.^2).*(X.^2))./W.^2);
            RmsX = sqrt(Z)*sqrt((sum((W.^2).*(X.^2)))./sum(W.^2));
            AvgX = 0;
            WinX = 0;
        otherwise
            AvgX = 0;
            RmsX = 0;
            WinX = 0;
    end
end

end

