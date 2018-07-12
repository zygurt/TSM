function [ Frame_width, File_width ] = st_width( input, N )
% [ Frame_width, File_width ] = st_width( input, N )
% Calculate the frame and file width
%   input is a 2 channel signal
%   N is the frame length.  Larger values give smoother output.
%   Frame_width is the width for each frame
%   File_width is the width for the entire file

% Tim Roberts - Griffith University 2018

% %Generate segmental RMS sum and difference vectors
frames(:,:,1) = buffer(input(:,1), N);
frames(:,:,2) = buffer(input(:,2), N);
input_sum_framed = frames(:,:,1)+frames(:,:,2);
input_diff_framed = frames(:,:,1)-frames(:,:,2);
sum_RMS = sqrt(mean(input_sum_framed.^2));
diff_RMS = sqrt(mean(input_diff_framed.^2));
Frame_width = zeros(1,length(sum_RMS));

for n = 1:length(sum_RMS)
    if (sum_RMS(n) ~= 0 && diff_RMS(n) == 0)
        Frame_width(n) = 1;
    elseif (sum_RMS(n) == 0 && diff_RMS(n) ~= 0)
        Frame_width(n) = -1;
    elseif (sum_RMS(n) == 0 && diff_RMS(n) == 0)
        Frame_width(n) = 0;
    else
        Frame_width(n) = 20*log10(sum_RMS(n)/diff_RMS(n));
    end
end

if(max(abs(Frame_width))~=0)
    Frame_width = Frame_width/max(abs(Frame_width));
end

File_width = mean(Frame_width);


% input_sum = input(:,1)+input(:,2);
% input_diff =  input(:,1)-input(:,2);
% 
% if max(max(abs([input_sum input_diff]))) ~= 0
%     input_sum=abs(input_sum)./max(max(abs([input_sum input_diff])));
%     input_diff=abs(input_diff)./max(max(abs([input_sum input_diff])));
%     res = (abs(input_sum)-abs(input_diff));
%     Frame_width = mean(buffer(res,N));
%     File_width = mean(Frame_width);
% else
%     %Signal is silence
%     Frame_width = 0;
%     File_width = 0;
% end


end

