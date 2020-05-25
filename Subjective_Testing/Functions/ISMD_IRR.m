function [ G, var ] = ISMD_IRR( OS, verbose )
%[ G ] = ISMD_IRR( Opinion_Scores )
%   Ill-Structured Measurement Designs Inter-rater Reliability
%   Based on Putka, D. and Le, H., Ill-Structured Measurement Designs in
%   Organizational Research: Implications for Estimating Enterrater
%   Reliability, Journal of Applied Psychology, Vol. 93, No. 5, 959-981 (2008)

% OS is the Opinion scores in a matrix with Files forming rows, Raters
% forming columns and ratings at the intersections.
% verbose allows for printing the location in the processing to terminal.

%Subsititute ratee for files for a  subjective quality of audio context

% k_hat is the harmonic mean of raters per ratee
% c is the number of raters that each pair of ratees share.
% kiki_prime are the number of raters who rated ratees i and i_prime respectively (i != i_prime)
% Nt is Total number of ratees

% load('Opinion_Matrix.mat');


% OS = OS_norm;
% if verbose
%     OS = [0 0 0 0 0 0 5 7 0 0 0 0 0;
%           0 0 0 0 4 0 0 0 2 0 0 0 0;
%           0 0 3 2 0 0 0 0 0 0 0 0 0;
%           5 0 0 0 0 4 0 0 0 0 0 0 0;
%           0 1 0 0 0 0 0 0 0 0 0 0 3;
%           0 7 0 0 0 0 0 0 0 0 0 5 0;
%           0 0 0 0 0 0 0 6 0 0 0 0 4;
%           0 0 0 0 5 3 0 0 0 0 0 0 0;
%           0 0 0 0 0 0 0 5 0 0 2 0 0];
% end
%Create a logical array of OS to enable summing and boolean operations
OS_v = OS~=0;
%Calculate the Harmonic Mean of the number of ratings per file
k_hat = harmmean(sum(OS_v,2));

%Calculate c and kiki_prime
%Swapped i and i_prime for a and a_prime to avoid complex
c = zeros(size(OS,1));
kiki_prime = zeros(size(OS,1));
for a=1:size(OS,1)
    if verbose
        if( mod(a,100)==0)
            fprintf('a = %d\n', a);
        end
    end
    for a_prime = a:size(OS,1) %Only need to calculate half of the matrix as it mirrors
        if a~=a_prime
            c(a,a_prime) = sum((OS_v(a,:) & OS_v(a_prime,:)));
            c(a_prime,a) = c(a,a_prime);
            kiki_prime(a,a_prime) = sum(OS_v(a,:))+sum(OS_v(a_prime,:));
            kiki_prime(a_prime,a) = kiki_prime(a,a_prime);
        end
    end
end
%Calculate the number of ratings per pair of files
Nt = size(OS,1);
%Remove the NaN values before summing due to diagonal zeros
temp = c./kiki_prime;
temp(isnan(temp)) = 0;
%Calculate the q scaling factor
q = 1/k_hat - sum(sum(temp))/(Nt*(Nt-1));


%Create the data structure required by the paper.
%Three columns, [File, Rater, Rating]
pos = 1;
d = zeros(sum(sum(OS_v)),3);
for n = 1:size(OS,1)
    for k = 1:size(OS,2)
        if OS(n,k)~=0
            d(pos,1) = n;
            d(pos,2) = k;
            d(pos,3) = OS(n,k);
            pos = pos+1;
        end
    end
end
tbl = table(d(:,1),d(:,2),d(:,3),'VariableNames',{'File', 'Rater', 'Rating'});
%Not really sure how to do the random effects model.
%SAS, SPSS and R code in the paper.

lme = fitlme(tbl,'Rating ~ 1 + (1|File) + (1|Rater)','FitMethod','REML');
% disp(lme)
% [B, Bnames, STATS] = randomEffects(lme, 'alpha',0.05);
% disp(STATS)
[psi,mse,~] = covarianceParameters(lme, 'alpha',0.05);

var_T = psi{1,1};
var_R = psi{2,1};
var_e = mse;
var = [var_T, var_R, var_e];

G = var_T/(var_T+(q*var_R+var_e/k_hat));


end

