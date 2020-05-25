function [ results ] = TSM_switch( TSM , results, m, n )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
switch (TSM)
    case '38.38'
        if isfield(results.scale.TSM38,'ave_all')
            results.scale.TSM38.ave_all = [results.scale.TSM38.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM38.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '44.27'
        if isfield(results.scale.TSM44,'ave_all')
            results.scale.TSM44.ave_all = [results.scale.TSM44.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM44.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '53.83'
        if isfield(results.scale.TSM53,'ave_all')
            results.scale.TSM53.ave_all = [results.scale.TSM53.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM53.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '65.24'
        if isfield(results.scale.TSM65,'ave_all')
            results.scale.TSM65.ave_all = [results.scale.TSM65.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM65.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '78.21'
        if isfield(results.scale.TSM78,'ave_all')
            results.scale.TSM78.ave_all = [results.scale.TSM78.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM78.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '82.58'
        if isfield(results.scale.TSM82,'ave_all')
            results.scale.TSM82.ave_all = [results.scale.TSM82.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM82.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '99.61'
        if isfield(results.scale.TSM99,'ave_all')
            results.scale.TSM99.ave_all = [results.scale.TSM99.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM99.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '138.1'
        if isfield(results.scale.TSM138,'ave_all')
            results.scale.TSM138.ave_all = [results.scale.TSM138.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM138.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '166.7'
        if isfield(results.scale.TSM166,'ave_all')
            results.scale.TSM166.ave_all = [results.scale.TSM166.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM166.ave_all = results.set(n).file(m).MOS_avg;
        end
    case '192.4'
        if isfield(results.scale.TSM192,'ave_all')
            results.scale.TSM192.ave_all = [results.scale.TSM192.ave_all results.set(n).file(m).MOS_avg];
        else
            results.scale.TSM192.ave_all = results.set(n).file(m).MOS_avg;
        end
    otherwise
        
end

end

