function [d] = cohen(ref,test)
M1 = mean(test);
M2 = mean(ref);
SD1 = std(test);
SD2 = std(ref);
SD_Pool = sqrt((SD1^2+SD2^2)/2);
%Calculate mean and standard error of Difference
diff = M1-M2;
d = diff/SD_Pool;
end