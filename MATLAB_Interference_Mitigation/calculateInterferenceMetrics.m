function metrics = calculateInterferenceMetrics(radar_cube, simulation_info)
    % Initialize metrics structure
    metrics = struct();
    num_frames = size(radar_cube, 3);
    
    metrics.INR = zeros(1, num_frames);          % Interference-to-Noise Ratio
    metrics.SIR = zeros(1, num_frames);          % Signal-to-Interference Ratio
    metrics.interference_power = zeros(1, num_frames);
    metrics.interference_bandwidth = zeros(1, num_frames);
    
    for frame = 1:num_frames
        % Extract components from simulation info
        clean_signal = simulation_info(frame).clean_signal;
        interference = simulation_info(frame).interference;
        noise = simulation_info(frame).noise;
        
        signal_power = mean(abs(clean_signal).^2, 'all');
        interference_power = mean(abs(interference).^2, 'all');
        noise_power = mean(abs(noise).^2, 'all');
        
        metrics.INR(frame) = 10 * log10(interference_power / noise_power);
        metrics.SIR(frame) = 10 * log10(signal_power / interference_power);
        metrics.interference_power(frame) = 10 * log10(interference_power);
        
        % Calculate interference bandwidth
        metrics.interference_bandwidth(frame) = calculateInterferenceBandwidth(interference);
    end
    
    metrics.average_INR = mean(metrics.INR);
    metrics.average_SIR = mean(metrics.SIR);
    metrics.peak_interference = max(metrics.interference_power);
    metrics.average_interference_power = mean(metrics.interference_power);

    if isfield(simulation_info(1).interference_info, 'direct')
        metrics.direct_interference = analyzeInterferenceType(simulation_info, 'direct');
    end
    if isfield(simulation_info(1).interference_info, 'multipath')
        metrics.multipath_interference = analyzeInterferenceType(simulation_info, 'multipath');
    end
    if isfield(simulation_info(1).interference_info, 'crosstalk')
        metrics.crosstalk_interference = analyzeInterferenceType(simulation_info, 'crosstalk');
    end
end

% Helper function to calculate interference bandwidth
function bandwidth = calculateInterferenceBandwidth(interference)
    % Calculate power spectrum
    spectrum = fft2(interference);
    power_spectrum = abs(spectrum).^2;
    
    % Calculate average power spectrum along Doppler dimension
    avg_spectrum = mean(power_spectrum, 2);
    
    % Find -3dB bandwidth
    max_power = max(avg_spectrum);
    threshold = max_power / 2;  % -3dB point
    
    above_threshold = avg_spectrum > threshold;
    bandwidth = sum(above_threshold) / length(avg_spectrum);
end

% Helper function to analyze specific interference types
function type_metrics = analyzeInterferenceType(simulation_info, interference_type)
    type_metrics = struct();
    num_frames = length(simulation_info);
    
    % Initialize arrays for type-specific metrics
    power_levels = [];
    temporal_characteristics = [];
    
    % Collect metrics across frames
    for frame = 1:num_frames
        if isfield(simulation_info(frame).interference_info, interference_type)
            info = simulation_info(frame).interference_info.(interference_type);
            
            % Extract available metrics based on interference type
            switch interference_type
                case 'direct'
                    if isfield(info, 'power')
                        power_levels = [power_levels, [info.power]];
                    end
                    if isfield(info, 'rel_velocity')
                        temporal_characteristics = [temporal_characteristics, [info.rel_velocity]];
                    end
                    
                case 'multipath'
                    if isfield(info, 'attenuation')
                        power_levels = [power_levels, [info.attenuation]];
                    end
                    if isfield(info, 'delay')
                        temporal_characteristics = [temporal_characteristics, [info.delay]];
                    end
                    
                case 'crosstalk'
                    if isfield(info, 'power')
                        power_levels = [power_levels, [info.power]];
                    end
                    if isfield(info, 'freq_offset')
                        temporal_characteristics = [temporal_characteristics, [info.freq_offset]];
                    end
            end
        end
    end
    
    if ~isempty(power_levels)
        type_metrics.average_power = mean(power_levels);
        type_metrics.max_power = max(power_levels);
        type_metrics.power_std = std(power_levels);
    end
    
    if ~isempty(temporal_characteristics)
        type_metrics.average_temporal = mean(temporal_characteristics);
        type_metrics.temporal_std = std(temporal_characteristics);
    end
    
    type_metrics.occurrence_rate = sum(~cellfun(@isempty, {simulation_info.interference_info})) / num_frames;
end
