function [range_idx, doppler_idx] = convertToIndices(range, velocity, config)
    range_res = config.radar.c / (2 * config.radar.bw);
    range_idx = round(range / range_res) + 1;
    
    velocity_res = config.radar.lambda / (2 * config.radar.tm * config.frame.num_chirps);
    doppler_idx = round(velocity / velocity_res) + config.frame.num_chirps/2;
    
    range_idx = max(1, min(range_idx, config.frame.num_samples));
    doppler_idx = max(1, min(doppler_idx, config.frame.num_chirps));
end
