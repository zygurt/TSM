%Anonymise results

%Set a seed
rng(1234);


%Anonymise individual results files
d = dir('./Results_All');
id_list = [];
for n = 1:length(d)
    if d(n).isdir == 0
        load(['./Results_All/' d(n).name]);
        %generate random identifier
        id = randi(2^16,1);
        %If id is in the list generate until a new one is found.
        while sum(id_list==id)>0
            %generate random identifier
            id = randi(2^16,1);
        end
        %Append to list of random values
        id_list = [id_list id];
        user_data.name = num2str(id);
        user_data.email = 'redacted';
        sname = sprintf('./Results_Anon_All/%s.mat',num2str(id));
        save(sname,'user_data','-v7')
    end
    
end

