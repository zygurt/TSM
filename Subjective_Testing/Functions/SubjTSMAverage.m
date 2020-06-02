function [Means,cats] = SubjTSMAverage(x,y,edges,categ)
%[Means] = SubjTSMAverage(x,y,edges)
%   This function calculates means of regions within a vector.
%   x is the original independent variable
%   y is the original dependent variable
%   edges is a vector containing edges of regions

if nargin<4
    categ(1:length(x)) = {'All'};
    cats = {'All'};
else
    cats = unique(categ);
    cats{length(cats)+1} = 'All';
end
k = 1;
cat_idx = zeros(size(categ));
for n = 1:length(categ)
    while ~(strcmp(categ{n},cats{k}))
        k = k+1;
    end
    cat_idx(n) = k;
    k = 1;
end
num_segments = length(edges)-1;
counts = zeros(length(cats),num_segments);
totals = zeros(length(cats),num_segments);

for n = 1:length(x)
    for k = 1:(length(edges)-1)
        if x(n)>edges(k) && x(n)<edges(k+1)
            totals(cat_idx(n),k) = totals(cat_idx(n),k)+y(n);
            counts(cat_idx(n),k) = counts(cat_idx(n),k)+1;
            totals(length(cats),k) = totals(length(cats),k)+y(n);
            counts(length(cats),k) = counts(length(cats),k)+1;
            break;
        end
    end
end


Means = totals./counts;
% Means(isnan(Mean)) = 
end

