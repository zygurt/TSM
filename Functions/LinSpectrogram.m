function [ handle ] = LinSpectrogram( x, Fs, ms, contrast )
% [ handle ] = LinSpectrogram( x, Fs, ms, contrast )
% Create a spectrogram with log scaled frequency
% The size is set to be useful for 2 column papers.
%  x = input waveform
%  Fs = waveform sampling frequency
%  ms = window period in ms
%  contrast = integer for progressive log scaling of color scale

% Tim Roberts - Griffith University 2018

N = 2^nextpow2(Fs*ms*10^(-3));
%Set Latex font
set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultAxesFontName', 'Times New Roman')
%Create new figure
handle = figure;
for c = 1:size(x,2)
    %Calculate Spectrogram
    [S,F,T] = spectrogram(x(:,c), N, 3*N/4, N, Fs,'yaxis');
    %Normalise S and compute power spectrogram
    S = 20*log10(abs(S)/max(max(abs(S))));
    %Plot the spectrogram
    subplot(size(x,2),1,c);
    surf(T,F,S,'EdgeColor','none');
    view([0 90])
    % axis tight
    axis([min(T) max(T) min(F) max(F)])
    %Turn grid lines off
    grid('off')
    %Labels
    xlabel('\textbf{Time(s)}')
    ylabel('\textbf{Frequency (Hz)}')
    if(size(x,2)>1)
        title_text = sprintf('Channel %d',c);
        title(title_text);
    end
    %Set the yticks and labels
    num_ticks = 10;
    tick_loc = linspace(0,Fs/2,num_ticks);
    yticks(tick_loc);
    yticklabels(floor(tick_loc));
    %Set the color map
    colormap gray
    map = colormap;
    for n = 1:contrast
        map = (map.*repmat(logspace(0,1,64),3,1)')/10;
    end
    colormap(1-map);

end
    %Set the size of the plot
    x0=10;
    y0=10;
    width=300;
    height=200*size(x,2);
    set(gcf,'units','points','position',[x0,y0,width,height])
end

