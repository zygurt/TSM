function [ results ] = method_TSM_switch( method, TSM , results, m, n )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
switch (TSM)
    case '38.38'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM38')
                    results.method.PV.TSM38 = [results.method.PV.TSM38 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM38 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM38')
                    results.method.IPL.TSM38 = [results.method.IPL.TSM38 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM38 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM38')
                    results.method.HPTSM.TSM38 = [results.method.HPTSM.TSM38 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM38 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM38')
                    results.method.WSOLA.TSM38 = [results.method.WSOLA.TSM38 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM38 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM38')
                    results.method.uTVS.TSM38 = [results.method.uTVS.TSM38 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM38 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM38')
                    results.method.FESOLA.TSM38 = [results.method.FESOLA.TSM38 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM38 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
        
    case '44.27'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM44')
                    results.method.PV.TSM44 = [results.method.PV.TSM44 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM44 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM44')
                    results.method.IPL.TSM44 = [results.method.IPL.TSM44 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM44 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM44')
                    results.method.HPTSM.TSM44 = [results.method.HPTSM.TSM44 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM44 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM44')
                    results.method.WSOLA.TSM44 = [results.method.WSOLA.TSM44 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM44 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM44')
                    results.method.uTVS.TSM44 = [results.method.uTVS.TSM44 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM44 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM44')
                    results.method.FESOLA.TSM44 = [results.method.FESOLA.TSM44 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM44 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case '53.83'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM53')
                    results.method.PV.TSM53 = [results.method.PV.TSM53 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM53 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM53')
                    results.method.IPL.TSM53 = [results.method.IPL.TSM53 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM53 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM53')
                    results.method.HPTSM.TSM53 = [results.method.HPTSM.TSM53 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM53 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM53')
                    results.method.WSOLA.TSM53 = [results.method.WSOLA.TSM53 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM53 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM53')
                    results.method.uTVS.TSM53 = [results.method.uTVS.TSM53 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM53 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM53')
                    results.method.FESOLA.TSM53 = [results.method.FESOLA.TSM53 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM53 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case '65.24'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM65')
                    results.method.PV.TSM65 = [results.method.PV.TSM65 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM65 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM65')
                    results.method.IPL.TSM65 = [results.method.IPL.TSM65 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM65 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM65')
                    results.method.HPTSM.TSM65 = [results.method.HPTSM.TSM65 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM65 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM65')
                    results.method.WSOLA.TSM65 = [results.method.WSOLA.TSM65 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM65 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM65')
                    results.method.uTVS.TSM65 = [results.method.uTVS.TSM65 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM65 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM65')
                    results.method.FESOLA.TSM65 = [results.method.FESOLA.TSM65 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM65 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case '78.21'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM78')
                    results.method.PV.TSM78 = [results.method.PV.TSM78 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM78 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM78')
                    results.method.IPL.TSM78 = [results.method.IPL.TSM78 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM78 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM78')
                    results.method.HPTSM.TSM78 = [results.method.HPTSM.TSM78 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM78 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM78')
                    results.method.WSOLA.TSM78 = [results.method.WSOLA.TSM78 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM78 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM78')
                    results.method.uTVS.TSM78 = [results.method.uTVS.TSM78 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM78 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM78')
                    results.method.FESOLA.TSM78 = [results.method.FESOLA.TSM78 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM78 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case '82.58'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM82')
                    results.method.PV.TSM82 = [results.method.PV.TSM82 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM82 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM82')
                    results.method.IPL.TSM82 = [results.method.IPL.TSM82 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM82 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM82')
                    results.method.HPTSM.TSM82 = [results.method.HPTSM.TSM82 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM82 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM82')
                    results.method.WSOLA.TSM82 = [results.method.WSOLA.TSM82 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM82 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM82')
                    results.method.uTVS.TSM82 = [results.method.uTVS.TSM82 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM82 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM82')
                    results.method.FESOLA.TSM82 = [results.method.FESOLA.TSM82 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM82 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case '99.61'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM99')
                    results.method.PV.TSM99 = [results.method.PV.TSM99 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM99 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM99')
                    results.method.IPL.TSM99 = [results.method.IPL.TSM99 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM99 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM99')
                    results.method.HPTSM.TSM99 = [results.method.HPTSM.TSM99 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM99 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM99')
                    results.method.WSOLA.TSM99 = [results.method.WSOLA.TSM99 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM99 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM99')
                    results.method.uTVS.TSM99 = [results.method.uTVS.TSM99 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM99 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM99')
                    results.method.FESOLA.TSM99 = [results.method.FESOLA.TSM99 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM99 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case '138.1'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM138')
                    results.method.PV.TSM138 = [results.method.PV.TSM138 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM138 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM138')
                    results.method.IPL.TSM138 = [results.method.IPL.TSM138 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM138 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM138')
                    results.method.HPTSM.TSM138 = [results.method.HPTSM.TSM138 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM138 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM138')
                    results.method.WSOLA.TSM138 = [results.method.WSOLA.TSM138 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM138 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM138')
                    results.method.uTVS.TSM138 = [results.method.uTVS.TSM138 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM138 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM138')
                    results.method.FESOLA.TSM138 = [results.method.FESOLA.TSM138 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM138 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case '166.7'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM166')
                    results.method.PV.TSM166 = [results.method.PV.TSM166 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM166 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM166')
                    results.method.IPL.TSM166 = [results.method.IPL.TSM166 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM166 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM166')
                    results.method.HPTSM.TSM166 = [results.method.HPTSM.TSM166 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM166 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM166')
                    results.method.WSOLA.TSM166 = [results.method.WSOLA.TSM166 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM166 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM166')
                    results.method.uTVS.TSM166 = [results.method.uTVS.TSM166 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM166 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM166')
                    results.method.FESOLA.TSM166 = [results.method.FESOLA.TSM166 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM166 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case '192.4'
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'TSM192')
                    results.method.PV.TSM192 = [results.method.PV.TSM192 results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.TSM192 = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'TSM192')
                    results.method.IPL.TSM192 = [results.method.IPL.TSM192 results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.TSM192 = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'TSM192')
                    results.method.HPTSM.TSM192 = [results.method.HPTSM.TSM192 results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.TSM192 = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'TSM192')
                    results.method.WSOLA.TSM192 = [results.method.WSOLA.TSM192 results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.TSM192 = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'TSM192')
                    results.method.uTVS.TSM192 = [results.method.uTVS.TSM192 results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.TSM192 = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'TSM192')
                    results.method.FESOLA.TSM192 = [results.method.FESOLA.TSM192 results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.TSM192 = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    otherwise
        disp('Unknown TSM ratio');
end

end

