function metrics = calculateValidationMetrics(radar_cube, simulation_info, config)
    metrics = struct();
    
    metrics.SINR = calculateSINR(radar_cube, simulation_info);
    
    metrics.detection = calculateDetectionMetrics(radar_cube, simulation_info, config);
    
    metrics.estimation = calculateEstimationAccuracy(radar_cube, simulation_info, config);
    
    metrics.interference = calculateInterferenceMetrics(radar_cube, simulation_info);
end


% Helper functions for validation metrics
function SINR = calculateSINR(radar_cube, simulation_info)
    % Calculate SINR for each frame
    num_frames = size(radar_cube, 3);
    SINR = zeros(1, num_frames);
    
    for frame = 1:num_frames
        signal_power = mean(abs(simulation_info(frame).clean_signal).^2, 'all');
        interference_power = mean(abs(simulation_info(frame).interference).^2, 'all');
        noise_power = mean(abs(simulation_info(frame).noise).^2, 'all');
        
        SINR(frame) = 10 * log10(signal_power / (interference_power + noise_power));
    end
end

function detection_metrics = calculateDetectionMetrics(radar_cube, simulation_info, config)
    % Initialize detection metrics
    detection_metrics = struct();
    num_frames = size(radar_cube, 3);
    detection_metrics.probability_detection = zeros(1, num_frames);
    detection_metrics.false_alarm_rate = zeros(1, num_frames);
    
    % Process each frame
    for frame = 1:num_frames
        % Apply CFAR detection
        [detections, threshold] = applyCFAR(radar_cube(:,:,frame), config);
        
        % Compare with ground truth
        [pd, fa] = compareDetections(detections, simulation_info(frame).targets, config);
        
        % Store metrics
        detection_metrics.probability_detection(frame) = pd;
        detection_metrics.false_alarm_rate(frame) = fa;
        
        % Store additional information if needed
        detection_metrics.threshold_map(:,:,frame) = threshold;
        detection_metrics.detection_map(:,:,frame) = detections;
    end
    
    % Calculate summary statistics
    detection_metrics.average_pd = mean(detection_metrics.probability_detection);
    detection_metrics.average_fa = mean(detection_metrics.false_alarm_rate);
    detection_metrics.min_pd = min(detection_metrics.probability_detection);
    detection_metrics.max_fa = max(detection_metrics.false_alarm_rate);
end


