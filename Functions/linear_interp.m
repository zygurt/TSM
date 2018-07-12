function [ x ] = linear_interp( x )
%[ y ] = linear_interp( x )
%   Replaces zeros left after assigning new time scale
%I should probably just write the new values
%Also I should allow this to work for arrays, not just vectors.
low = 0;
low_n = 0;

count = 0;
for n = 1:length(x)
    if(x(n)~=0) && (count == 0)
        low = x(n);
        low_n = n;
        count = count+1;
    elseif(x(n)~=0 && count > 0)
        high = x(n);
        high_n = n;
        count = count+1;
        if(low == 0) 
            disp('Initial Value = 0');
            low = 0;
            low_n = 1;
        end
        x(low_n:high_n) = linspace(low, high,count); 
        low = high;
        low_n = high_n;
        count = 1;
    else
        count = count+1;
    end
end


end

