function radar_constraints = calculateRadarConstraints(config)
    % Range resolution
    range_res = config.radar.c / (2 * config.radar.bw);
    
    % Velocity resolution
    vel_res = config.radar.lambda / (2 * config.radar.tm * config.frame.num_chirps);
    
    % Maximum unambiguous range
    R_max = config.radar.c * config.radar.fs / (2 * config.chirp.slope);
    
    % Maximum unambiguous velocity
    v_max = config.radar.lambda / (4 * config.radar.tm);

    % Maximum beat(range) frequency
    beat_freq_max = 2*150*config.chirp.slope/config.radar.c;      % ~10 MHz < fs/2

    % Max Doppler frequency(shift)
    fd_max = 2*v_max/config.radar.lambda;
    % For v_max=72.2m/s, lambda=c/fc=3e8/77e9
    % the maximum fd should be 2*72.2/(3e8/77e9) ≈ 37kHz

    % Maximum received IF (IF bandwidth) (Hz)
    fif_max = (beat_freq_max + fd_max);

    % Required fs > 2*(7.813e6 + 37e3)
    fs_required = 2*(fif_max);
    
    % Store constraints
    radar_constraints.range_res = range_res;
    radar_constraints.vel_res = vel_res;
    radar_constraints.R_max = R_max;
    radar_constraints.v_max = v_max;
    
    % Print constraints for verification
    fprintf('\nRadar Constraints:\n');
    fprintf('Range Resolution: %.2f m\n', range_res);
    fprintf('Velocity Resolution: %.2f m/s\n', vel_res);
    fprintf('Maximum Range: %.2f m\n', R_max);
    fprintf('Maximum Velocity: %.2f m/s (%.2f km/h)\n', v_max, v_max*3.6);
    fprintf('Maximum IF frequency/bandwidth (range+doppler): %.2f MHz\n', fif_max/10e6);
    fprintf('Required sampling rate fs (Nyquist): %.2f MHz\n', fs_required/10e6);
end
