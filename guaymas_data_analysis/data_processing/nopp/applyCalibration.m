function NOPP = applyCalibration(cal)
% Takes in calibration struct produced by makeCalibration and apply
% respective calibration to user selected data. Saves data in a .mat
% structure that includes raw data, calibration, and calibrated data.

    [laser,aux]=importNopp();

    NOPP.cal = cal;
    NOPP.laser = laser;
    NOPP.aux = aux;
    
    NOPP.t = laser.t;
    NOPP.CH4_F = interp1(NOPP.cal.fPoints,NOPP.cal.ref,NOPP.laser.F);
    NOPP.CH4_R = interp1(NOPP.cal.rPoints,NOPP.cal.ref,NOPP.laser.R);

end