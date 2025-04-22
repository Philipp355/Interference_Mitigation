function [filtered_cube, total_response] = applyFIR(radar_cube, config, vehicles)
    % First design filter in range dimension
    fs = config.radar.fs;  
    [num_samples, num_chirps, num_frames] = size(radar_cube);
    
    beat_freqs = zeros(1, length(vehicles.targets));
    fprintf('Expected beat frequencies:\n');
    for i = 1:length(vehicles.targets)
        range = abs(vehicles.targets(i).range);
        beat_freqs(i) = 2 * range * config.chirp.slope / config.radar.c;
        fprintf('Target %d (range %.1fm): %.2f MHz\n', i, range, beat_freqs(i)/1e6);
    end
    
    % bandpass filters
    filter_order = 64;
    bandwidth = 2e6;  % 2 MHz bandwidth
    total_response = zeros(filter_order+1, 1);
    filter_responses = zeros(filter_order+1, length(beat_freqs));
    
    % apply multiple bandpass filters
    filtered_cube = zeros(size(radar_cube));
    filtered_cube = radar_cube;
    
    for i = 1:length(beat_freqs)
        % normalized frequencies
        f_center = beat_freqs(i) / fs;
        f_width = bandwidth / fs;
        f1 = f_center - f_width/2;
        f2 = f_center + f_width/2;
        
        if f1 > 0 && f2 < 1
            % Design individual bandpass filter
            bp_filter = fir1(filter_order, [f1 f2], 'bandpass');
            filter_responses(:,i) = bp_filter;
            fprintf('Added filter for %.2f MHz\n', beat_freqs(i)/1e6);
            
            for frame = 1:num_frames
                for chirp = 1:num_chirps
                    filtered_cube(:,chirp,frame) = filtfilt(bp_filter, 1, filtered_cube(:,chirp,frame));
                end
            end
            
            fprintf('Applied filter for %.2f MHz\n', beat_freqs(i)/1e6);
        end
    end
    
    % Sum responses for filtering
    total_response = sum(filter_responses, 2);
end