function analyzeBeatFrequencies(signal, config)
    figure('Name', 'Beat Frequency Analysis', 'Position', [100, 100, 1200, 800]);
    
    % Improved single chirp spectrum
    subplot(2,2,1);
    window = blackman(config.frame.num_samples);
    chirp_fft = fft(signal(:,1).*window);
    f = (0:config.frame.num_samples-1) * config.radar.fs/config.frame.num_samples;
    f = f(1:config.frame.num_samples/2);
    chirp_fft = chirp_fft(1:config.frame.num_samples/2);
    
    plot(f/1e6, 20*log10(abs(chirp_fft)/max(abs(chirp_fft))));
    title('Single Chirp Spectrum');
    xlabel('Frequency (MHz)');
    ylabel('Magnitude (dB)');
    grid on;
    ylim([-60 0]);


    subplot(2,2,2);
    % 2D FFT with proper windowing
    win2d = window * blackman(config.frame.num_chirps)';
    rd_map = fftshift(fft2(signal .* win2d));
    rd_map_db = 20*log10(abs(rd_map)/max(abs(rd_map(:))));
    
    velocity_axis = (-config.frame.num_chirps/2:config.frame.num_chirps/2-1) * ...
        config.radar.lambda/(2*config.radar.tm*config.frame.num_chirps);
    range_axis = (0:config.frame.num_samples-1) * config.radar.c/(2*config.radar.bw);
    
    imagesc(velocity_axis, range_axis, rd_map_db);
    title('Range-Velocity Map');
    xlabel('Velocity (m/s)');
    ylabel('Range (m)');
    colorbar;
    clim([-40 0]);
    
    % Phase stability
    subplot(2,2,[3,4]);
    phase_stability = angle(signal(round(config.frame.num_samples/2),:));
    plot(1:config.frame.num_chirps, unwrap(phase_stability), 'LineWidth', 1.5);
    title('Phase Stability Across Chirps');
    xlabel('Chirp Number');
    ylabel('Phase (rad)');
    grid on;
end

