function analyzeTargetResponse(rd_map, range_axis, velocity_axis, config)
    figure('Name', 'Target Response Analysis');
    
    % Plot full RD map
    subplot(2,2,1);
    rd_db = 20*log10(abs(rd_map));
    rd_db = rd_db - max(rd_db(:));
    imagesc(velocity_axis, range_axis, rd_db);
    title('Range-Doppler Map');
    xlabel('Velocity (m/s)');
    ylabel('Range (m)');
    clim([-40 0]);
    colorbar;
    
    % Find strongest peak
    [max_val, max_idx] = max(abs(rd_map(:)));
    [peak_range_idx, peak_vel_idx] = ind2sub(size(rd_map), max_idx);
    
    % Extract range profile around peak
    range_slice = rd_db(:, peak_vel_idx);
    subplot(2,2,2);
    plot(range_axis, range_slice);
    title('Range Profile through Peak');
    xlabel('Range (m)');
    ylabel('Magnitude (dB)');
    grid on;
    
    % Zoom in
    zoom_range = 20;  % meters
    peak_range = range_axis(peak_range_idx);
    range_mask = abs(range_axis - peak_range) < zoom_range;
    subplot(2,2,3);
    plot(range_axis(range_mask), range_slice(range_mask));
    title('Zoomed Range Profile');
    xlabel('Range (m)');
    ylabel('Magnitude (dB)');
    grid on;
    
    % 3dB width
    half_power = range_slice(peak_range_idx) - 3;
    width_mask = range_slice > half_power & range_mask;
    if any(width_mask)
        range_width = range_axis(find(width_mask, 1, 'last')) - range_axis(find(width_mask, 1, 'first'));
        fprintf('Target -3dB width: %.3f m\n', range_width);
        fprintf('Compared to range resolution: %.3f m\n', config.radar.c/(2*config.radar.bw));
    end
end
