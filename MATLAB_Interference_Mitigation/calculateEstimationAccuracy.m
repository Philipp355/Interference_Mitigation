function estimation_metrics = calculateEstimationAccuracy(radar_cube, simulation_info, config)
    num_frames = size(radar_cube, 3);
    estimation_metrics = struct();
    estimation_metrics.range_rmse = zeros(1, num_frames);
    estimation_metrics.velocity_rmse = zeros(1, num_frames);
    estimation_metrics.range_bias = zeros(1, num_frames);
    estimation_metrics.velocity_bias = zeros(1, num_frames);
    
    % Process each frame
    for frame = 1:num_frames
        [range_estimates, velocity_estimates, ~, ~] = detectTargets(radar_cube(:,:,frame), config);
        
        [range_error, velocity_error] = calculateErrors(range_estimates, ...
            velocity_estimates, simulation_info(frame).targets);
        
        % Calculate metrics for valid measurements
        valid_range = ~isnan(range_error);
        valid_velocity = ~isnan(velocity_error);
        
        if any(valid_range)
            estimation_metrics.range_rmse(frame) = sqrt(mean(range_error(valid_range).^2));
            estimation_metrics.range_bias(frame) = mean(range_error(valid_range));
        end
        
        if any(valid_velocity)
            estimation_metrics.velocity_rmse(frame) = sqrt(mean(velocity_error(valid_velocity).^2));
            estimation_metrics.velocity_bias(frame) = mean(velocity_error(valid_velocity));
        end
    end
    
    % average metrics
    estimation_metrics.average_range_rmse = mean(estimation_metrics.range_rmse);
    estimation_metrics.average_velocity_rmse = mean(estimation_metrics.velocity_rmse);
    estimation_metrics.average_range_bias = mean(estimation_metrics.range_bias);
    estimation_metrics.average_velocity_bias = mean(estimation_metrics.velocity_bias);
end


