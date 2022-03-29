function [laser,aux] = importNopp(filepath)
    %% importing NOPP datafile to Matlab
    % if file path is not defined exsplicitly prompt user to select
    if(exist('filepath','var')==0)
        [file,path] = uigetfile('*.txt');
        filepath    = fullfile(path,file);
    end
    % read filepath as table
    data = readtable(filepath);
    %% parse laser and auxilary data structures
    % laser: time(t), temperature(T), ringdown(R), fundamental(F)
    laser = data(:,{'Var1','Var2','Var3','Var4'});
    laser.Properties.VariableNames = {'t','T','R','F'};
    laser.Properties.VariableUnits = {'ISO' 'Celsius' 'PSI' ''};
    % auxillary: time(t), cell-pressure(cP), cell-temperature(cT),
    %            housing-pressure(hP), housting-temperature(hT),
    %            housting-relative-humidity(RH), power-states(states)
    aux   = data(:,{'Var5','Var6','Var7','Var8','Var9','Var10','Var11'});
    aux.Properties.VariableNames = {'t','cT','cP','hT','hP','RH','states'};
    aux.Properties.VariableUnits = {'ISO' 'Celsius' 'PSI' 'Celcius' 'PSI' '%' ''};
end