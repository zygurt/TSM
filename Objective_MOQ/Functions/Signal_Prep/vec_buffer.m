function [ x_buf ] = vec_buffer( x, N, frame_starts )
%[ x_buf ] = vec_buffer( x, N, frame_starts )
%   Buffers a vector input signal into frames of length N starting at
%   frame_starts
x_buf = zeros(N,length(frame_starts));
if frame_starts(end)+N > length(x)
    x = [x;zeros(frame_starts(end)+N-length(x),1)];
end
for n = 1:length(frame_starts)
    x_buf(:,n) = x(frame_starts(n):frame_starts(n)+N-1);
end


end

