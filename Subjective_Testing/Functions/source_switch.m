function [ results ] = source_switch( type, method, results, m, n )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
switch(type)
    case 'Complex'
        if isfield(results.type.complex,'ave_all')
            results.type.complex.ave_all = [results.type.complex.ave_all results.set(n).file(m).MOS_avg];
        else
            results.type.complex.ave_all = results.set(n).file(m).MOS_avg;
        end
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'complex')
                    results.method.PV.complex = [results.method.PV.complex results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.complex = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'complex')
                    results.method.IPL.complex = [results.method.IPL.complex results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.complex = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'complex')
                    results.method.HPTSM.complex = [results.method.HPTSM.complex results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.complex = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'complex')
                    results.method.WSOLA.complex = [results.method.WSOLA.complex results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.complex = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'complex')
                    results.method.uTVS.complex = [results.method.uTVS.complex results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.complex = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'complex')
                    results.method.FESOLA.complex = [results.method.FESOLA.complex results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.complex = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
    case 'Music'
        if isfield(results.type.music,'ave_all')
            results.type.music.ave_all = [results.type.music.ave_all results.set(n).file(m).MOS_avg];
        else
            results.type.music.ave_all = results.set(n).file(m).MOS_avg;
        end
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'music')
                    results.method.PV.music = [results.method.PV.music results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.music = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'music')
                    results.method.IPL.music = [results.method.IPL.music results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.music = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'music')
                    results.method.HPTSM.music = [results.method.HPTSM.music results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.music = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'music')
                    results.method.WSOLA.music = [results.method.WSOLA.music results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.music = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'music')
                    results.method.uTVS.music = [results.method.uTVS.music results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.music = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'music')
                    results.method.FESOLA.music = [results.method.FESOLA.music results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.music = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
        
    case 'Solo'
        if isfield(results.type.solo,'ave_all')
            results.type.solo.ave_all = [results.type.solo.ave_all results.set(n).file(m).MOS_avg];
        else
            results.type.solo.ave_all = results.set(n).file(m).MOS_avg;
        end
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'solo')
                    results.method.PV.solo = [results.method.PV.solo results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.solo = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'solo')
                    results.method.IPL.solo = [results.method.IPL.solo results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.solo = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'solo')
                    results.method.HPTSM.solo = [results.method.HPTSM.solo results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.solo = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'solo')
                    results.method.WSOLA.solo = [results.method.WSOLA.solo results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.solo = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'solo')
                    results.method.uTVS.solo = [results.method.uTVS.solo results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.solo = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'solo')
                    results.method.FESOLA.solo = [results.method.FESOLA.solo results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.solo = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
        
    case 'Voice'
        if isfield(results.type.voice,'ave_all')
            results.type.voice.ave_all = [results.type.voice.ave_all results.set(n).file(m).MOS_avg];
        else
            results.type.voice.ave_all = results.set(n).file(m).MOS_avg;
        end
        switch(method)
            case 'PV'
                if isfield(results.method.PV,'voice')
                    results.method.PV.voice = [results.method.PV.voice results.set(n).file(m).MOS_avg];
                else
                    results.method.PV.voice = results.set(n).file(m).MOS_avg;
                end
            case 'IPL'
                if isfield(results.method.IPL,'voice')
                    results.method.IPL.voice = [results.method.IPL.voice results.set(n).file(m).MOS_avg];
                else
                    results.method.IPL.voice = results.set(n).file(m).MOS_avg;
                end
            case 'HPTSM'
                if isfield(results.method.HPTSM,'voice')
                    results.method.HPTSM.voice = [results.method.HPTSM.voice results.set(n).file(m).MOS_avg];
                else
                    results.method.HPTSM.voice = results.set(n).file(m).MOS_avg;
                end
            case 'WSOLA'
                if isfield(results.method.WSOLA,'voice')
                    results.method.WSOLA.voice = [results.method.WSOLA.voice results.set(n).file(m).MOS_avg];
                else
                    results.method.WSOLA.voice = results.set(n).file(m).MOS_avg;
                end
            case 'uTVS'
                if isfield(results.method.uTVS,'voice')
                    results.method.uTVS.voice = [results.method.uTVS.voice results.set(n).file(m).MOS_avg];
                else
                    results.method.uTVS.voice = results.set(n).file(m).MOS_avg;
                end
            case 'FESOLA'
                if isfield(results.method.FESOLA,'voice')
                    results.method.FESOLA.voice = [results.method.FESOLA.voice results.set(n).file(m).MOS_avg];
                else
                    results.method.FESOLA.voice = results.set(n).file(m).MOS_avg;
                end
            otherwise
                disp('Method not known');
        end
        
    otherwise
        disp('Method not known');
        
end

end

