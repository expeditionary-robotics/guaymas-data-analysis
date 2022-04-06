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
    
    % time format
    infmt = 'yyyyMMdd''T''HHmmss.SSS';
    
    % laser: time(t), temperature(T), ringdown(R), fundamental(F)
    laser = data(:,{'Var1','Var2','Var3','Var4'});                      
    laser.Properties.VariableNames = {'t','T','R','F'};
    laser.Properties.VariableUnits = {'datetime' 'Celsius' 'PSI' ''};
    laser.t = datetime(laser.t,'InputFormat',infmt);    %datetime conversion
    
    % auxillary: time(t), cell-pressure(cP), cell-temperature(cT),
    %            housing-pressure(hP), housting-temperature(hT),
    %            housting-relative-humidity(RH), power-states(states)
    aux   = data(:,{'Var5','Var6','Var7','Var8','Var9','Var10','Var11'});
    aux.Properties.VariableNames = {'t','cT','cP','hT','hP','RH','states'};
    aux.Properties.VariableUnits = {'datetime' 'Celsius' 'PSI' 'Celcius' 'PSI' '%' ''};
    aux.t = datetime(aux.t,'InputFormat',infmt);    %datetime conversion

    % filtering outliers
    laser.F(laser.F>1)=nan;
    laser.R(laser.R>5)=nan;

end