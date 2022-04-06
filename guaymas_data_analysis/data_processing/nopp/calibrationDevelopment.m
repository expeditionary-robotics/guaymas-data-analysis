%% Produce a calibration
clear

% relative air calibration file path
path = "data/cal/air";
file = "N2cal.txt";

% reference gas values [PPM]
ref = [0 4 10 100 1000 10000];

cal = makeCalibration(path,file,ref);

%% applying calibration

NOPP = applyCalibration(cal);

%% plot data;

figure()
    plot(NOPP.t,movmean(NOPP.CH4_F,10),NOPP.t,movmean(NOPP.CH4_R,10));yyaxis right;plot(NOPP.t,NOPP.laser.F)
