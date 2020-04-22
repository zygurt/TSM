function [ x ] = linear_interp_zeros( x, TSM )
%[ y ] = linear_interp_zeros( x, TSM )
%   Replaces zeros left after assigning new time scale in uTVS time scaling
%   method.
n = find(x);
n_ = find(x==0);
if(TSM<0.5)
    for k = 1:length(n)-1
        x(n(k):n(k+1)) = linspace(x(n(k)),x(n(k+1)),n(k+1)-n(k)+1);
    end
else
    for k = 1:length(n_)-1
        if((n_(k)-1)>=1) %Avoid first sample
            %Find the next non-zero sample
            q = n_(k);
            while(x(q)==0)
                q = q+1;
            end
            x(n_(k)) = (x(n_(k)-1)+x(q))/2;
        end
    end
end


end

