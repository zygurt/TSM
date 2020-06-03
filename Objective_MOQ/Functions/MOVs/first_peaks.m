function [ peaks ] = first_peaks( MAG )
%[peaks] = first_peaks( MAG )
%   Finds the first peak within an array
%   A peaks is greater than the surrounding 4 samples

zp_MAG = [zeros(2, size(MAG,2)) ; MAG ; zeros(2, size(MAG,2))];
peaks = zeros(1,size(MAG,2));
valleys = zeros(1,size(MAG,2));
%Find the valleys
fp = 1;
k = 3;
for n = 1:size(zp_MAG,2)
    while fp && k<size(zp_MAG,1)-2
        if zp_MAG(k,n)<zp_MAG(k-2,n) && zp_MAG(k,n)<zp_MAG(k-1,n) && zp_MAG(k,n)<zp_MAG(k+1,n) && zp_MAG(k,n)<zp_MAG(k+2,n)
            valleys(n) = k-1;
            fp = 0;
        end
        k = k+1;
    end
    fp=1;
    k = 3;
end
%Find the next peak
fp = 1;
k = 3;
for n = 1:size(zp_MAG,2)
    while fp && k<size(zp_MAG,1)-2
        if zp_MAG(k,n)>zp_MAG(k-2,n) && zp_MAG(k,n)>zp_MAG(k-1,n) && zp_MAG(k,n)>zp_MAG(k+1,n) && zp_MAG(k,n)>zp_MAG(k+2,n) && k>valleys(n)
            peaks(n) = k-1;
            fp = 0;
        end
        k = k+1;
    end
    fp=1;
    k = 3;
end


end

