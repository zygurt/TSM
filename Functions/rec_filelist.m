function [ filelist ] = rec_filelist( f )
%[ filelist ] = rec_filelist( f )
%   Creates a list of all files within folder 'f' including subfolders 
%   Returns a filelist

%Create a string containing all the folders
allSubFolders = genpath(f);
% Parse into a cell array.
remain = allSubFolders;
listOfFolderNames = {};
%Break the string into individual folder names
while true
    [singleSubFolder, remain] = strtok(remain, pathsep);
    if isempty(singleSubFolder)
        break;
    end
    listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames);

%Create list of result files
file_count = 1;
for k = 1:numberOfFolders
    directory = dir(listOfFolderNames{k});
    for l = 1:numel(directory)
        if directory(l).isdir == 0
            final_char = char(listOfFolderNames{k});
            final_char = final_char(end);
            if(final_char == '/')
                res_filelist{file_count} = sprintf('%s%s',listOfFolderNames{k},directory(l).name);
                file_count = file_count+1;
            else
                res_filelist{file_count} = sprintf('%s/%s',listOfFolderNames{k},directory(l).name);
                file_count = file_count+1;
            end
        end
    end
end

filelist = res_filelist';

end

