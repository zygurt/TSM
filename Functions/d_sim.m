function [ Y ] = d_sim( A, B )
%[ Y ] = d_sim( A, B )
%Mean Euclidean distance between 2 horizontal vectors A-B
%  Vectors are padded to be the same length

%Orient in the same direction
if(size(A,1)>size(A,2))
    A = A';
end
if(size(B,1)>size(B,2))
    B = B';
end

%Match the lengths
A_length = length(A);
B_length = length(B);
if A_length ~= B_length
    if A_length > B_length
        B = [B zeros(1, A_length - B_length)];
    else
        A = [A zeros(1, B_length - A_length)];
    end
end

%Compute the mean Euclidean distance
Y = mean(sqrt((A-B).^2));

end

