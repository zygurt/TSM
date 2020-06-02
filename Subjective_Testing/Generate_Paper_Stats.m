%Paper Stats Generation

addpath('../Functions/');
audioIn = 'Source/';

source_filelist = rec_filelist(audioIn);

for n = 1:length(source_filelist)
    [x,fs] = audioread(char(source_filelist(n)));
    audio_data(n).name = source_filelist(n);
    audio_data(n).duration = length(x)/fs;
    audio_data(n).Lp = 20*log10(rms(x)/(20*10^-6));
end
fid = fopen('log_Anon.txt','a');
fprintf(fid,'\n Dataset stats for paper.\n');
fprintf(fid,'Mean file duration: %gs\n',mean([audio_data.duration]));
fprintf(fid,'Std file duration: %gs\n',std([audio_data.duration]));
fprintf(fid,'Min SPL: %gdB\n',min([audio_data.Lp]));
fprintf(fid,'Max SPL: %gdB\n',max([audio_data.Lp]));
fprintf(fid,'Mean SPL: %gdB\n',mean([audio_data.Lp]));
fprintf(fid,'STD SPL: %gdB\n',std([audio_data.Lp]));
fclose(fid);
