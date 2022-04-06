function cal = makeCalibration(path,file)
    %% import
    filepath = fullfile(path,file);
    % import cal for NOPP1
    [laser, aux] = importNopp(filepath);
    %% plotting
    
    % reference gas values [PPM]
    ref = [0 4 10 100 1000 10000];
    
    % plot time series
    figure('units','normalized','outerposition',[0 0 1 1])  % full screen
        subplot(2,1,1)
            plot(movmean(laser.R,10));hold on;yyaxis right;plot(movmean(laser.F,10))
    
    
    if(exist('fPoints','var')==0)
        % user select equilibration input
        for i=1:length(ref)
        
            [x(:,i),~] = ginput(2)
            T=x(1,i):x(2,i);
            T=round(T);
            
            subplot(2,1,1);hold on
                yyaxis left;plot(T,laser.R(T),'k');yyaxis right;plot(T,laser.F(T),'k')
        
            % calculate average
            fPoints(i) = mean(laser.F(T),'omitnan');
            rPoints(i) = mean(laser.R(T),'omitnan');
    
            subplot(2,2,3);hold on;bubblechart(fPoints(i),log10(ref(i)),1,"black")
            subplot(2,2,4);hold on;bubblechart(rPoints(i),log10(ref(i)),1,"black")
        
        end
    
    else
    
        subplot(2,2,3);hold on;bubblechart(fPoints,log10(ref),1,"black")
        subplot(2,2,4);hold on;bubblechart(rPoints,log10(ref),1,"black")
    
    end 
    
    %% Organize calibration into structure
    cal.ref = ref;
    cal.rPoints = rPoints;
    cal.fPoints= fPoints;
    cal.aux = aux;
    cal.laser = laser;
    
    %% save calibration in origonal directory
    tempStr = split(file,'.');
    filepath = fullfile(path,strcat(tempStr(1),".mat"));
    save(filepath, '-struct', 'cal');
end