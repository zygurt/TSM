function [ results ] = method_switch( method, results, m, n )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
switch(method)
    case 'PV'
        if isfield(results.method.PV,'ave_all')
            results.method.PV.ave_all = [results.method.PV.ave_all results.set(n).file(m).MOS_avg];
        else
            results.method.PV.ave_all = results.set(n).file(m).MOS_avg;
        end
    case 'IPL'
        if isfield(results.method.IPL,'ave_all')
            results.method.IPL.ave_all = [results.method.IPL.ave_all results.set(n).file(m).MOS_avg];
        else
            results.method.IPL.ave_all = results.set(n).file(m).MOS_avg;
        end
    case 'HPTSM'
        if isfield(results.method.HPTSM,'ave_all')
            results.method.HPTSM.ave_all = [results.method.HPTSM.ave_all results.set(n).file(m).MOS_avg];
        else
            results.method.HPTSM.ave_all = results.set(n).file(m).MOS_avg;
        end
    case 'WSOLA'
        if isfield(results.method.WSOLA,'ave_all')
            results.method.WSOLA.ave_all = [results.method.WSOLA.ave_all results.set(n).file(m).MOS_avg];
        else
            results.method.WSOLA.ave_all = results.set(n).file(m).MOS_avg;
        end
    case 'uTVS'
        if isfield(results.method.uTVS,'ave_all')
            results.method.uTVS.ave_all = [results.method.uTVS.ave_all results.set(n).file(m).MOS_avg];
        else
            results.method.uTVS.ave_all = results.set(n).file(m).MOS_avg;
        end
    case 'FESOLA'
        if isfield(results.method.FESOLA,'ave_all')
            results.method.FESOLA.ave_all = [results.method.FESOLA.ave_all results.set(n).file(m).MOS_avg];
        else
            results.method.FESOLA.ave_all = results.set(n).file(m).MOS_avg;
        end
    otherwise
        disp('Method not known');
        
end

end

