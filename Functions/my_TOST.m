function [H,pL,pU] = my_TOST(ref,test,vars)
%[H] = my_TOST(ref,test,vars)
%     ref is vector of reference observations
%     test is a vector of test observations
%     vars contains all the side settings
%     vars.alpha is the confidence level. Default: 0.05
%     vars.theta is the equivalence interval. Taken as literal input, unless percent_flag is set
%         then it is taken as a percentage of the reference mean. Default: 5% of Reference mean.
%     vars.percent_flag sets if theta is taken as a percentage of the reference mean. Default: 1.
%     vars.equal_var_flag sets if the two inputs samples have identical variance. Default: 0.
%     vars.plot_flag turns CI plotting on. Default: 0.
%     vars.title_name is the title given to the plot. Default: ''
%   Implemented using equations in:
%   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5502906/
%   and https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full
%   and checked against Two-Sample Equivalence Test in Minitab
%   Uses (1-alpha)100% confidence interval
%   Tim Roberts 2020 Griffith University

if nargin <3
    alpha = 0.05;
    theta = 0.05;
    percent_flag = 1;
    equal_var_flag = 0;
    plot_flag = 0;
    title_name = '';
else
    alpha = vars.alpha;
    theta = vars.theta;
    percent_flag = vars.percent_flag;
    equal_var_flag = vars.equal_var_flag;
    plot_flag = vars.plot_flag;
    title_name = vars.title_name;
end

M1 = mean(test);
M2 = mean(ref);
SD1 = std(test);
SD2 = std(ref);
n1 = length(test);
n2 = length(ref);
SE1 = SD1/sqrt(n1);
SE2 = SD2/sqrt(n2);

%Set equivalent interval to percentage of Reference Mean
if percent_flag
    equiv = theta*M2;
else
    equiv = theta;
end
%Calculate mean and standard error of Difference
diff = M1-M2;
SE = sqrt(SE1^2+SE2^2);

%Calculate Confidence Interval
dfw = (SD1^2/n1+SD2^2/n2)^2/ ...
      ((SD1^2/n1)^2/(n1-1)+(SD2^2/n2)^2/(n2-1));
TL = tinv(alpha,dfw); %(1-alpha)100% CI
% TL = tinv(2*alpha,dfw); %(1-2*alpha)100% CI
CI = TL*SE;
CIL = diff-CI;
CIU = diff+CI;

if plot_flag
    %Plot the confidence interval
    figure
    line([-equiv -equiv],[0 1],'Color','blue')
    line([equiv equiv],[0 1],'Color','blue')
    line([CIL CIL],[0.3 0.7],'Color','red','LineStyle','--')
    line([CIU CIU],[0.3 0.7],'Color','red','LineStyle','--')
    line([CIU CIL],[0.5 0.5],'Color','red','LineStyle','--')
    title(title_name)
    yticks({})
    axis([1.1*min([-equiv,CIU]), 1.1*max([equiv,CIL]), 0 1])
end

%Set equivalence claim
if CIU>=-equiv && CIL<=equiv
    H = 1;
%     fprintf('Can claim equivalence based on %g\% confidence interval for equivalence.\n',(1-alpha)*100)
else
    H = 0;
%     fprintf('Cannot claim equivalence based on %g\% confidence interval for equivalence.\n',(1-alpha)*100)
end

%Calculate t scores
if equal_var_flag
%Student's t test assuming equal variances
    sigma = sqrt(((n1-1)*SD1^2+(n2-1)*SD2^2)/(n1+n2-2)); %Pooled std
    tL = (M1-M2-(-equiv))/(sigma*sqrt(1/n1+1/n2));
    tU = (M1-M2-equiv)/(sigma*sqrt(1/n1+1/n2));
else
%Welch's t test assuming unequal variances
    tL = (M1-M2-(-equiv))/(sqrt(SD1^2/n1+SD2^2/n2));
    tU = (M1-M2-equiv)/(sqrt(SD1^2/n1+SD2^2/n2));
end

%Calculate p scores
pL = 1-tcdf(tL,n1+n2-2);
pU = tcdf(tU,n1+n2-2);

end

