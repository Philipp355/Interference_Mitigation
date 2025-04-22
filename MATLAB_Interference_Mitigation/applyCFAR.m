function [detections, threshold] = applyCFAR(rd_map)
    % CFAR parameters
    guard_cells = 2;      % Guard cells each side
    training_cells = 4;   % Training cells each side
    pfa = 1e-3;          % False alarm rate
    
    % Get magnitude
    rd_mag = abs(rd_map);
    rd_mag = rd_mag / max(rd_mag(:));
    
    threshold = zeros(size(rd_mag));
    detections = false(size(rd_mag));
    
    num_training = training_cells * 2;
    alpha = num_training * (pfa^(-1/num_training) - 1);
    
    % Apply 2D CA-CFAR
    for range_idx = 1+training_cells+guard_cells : size(rd_mag,1)-training_cells-guard_cells
        for doppler_idx = 1+training_cells+guard_cells : size(rd_mag,2)-training_cells-guard_cells
            % Extract CUT
            cut = rd_mag(range_idx, doppler_idx);
            
            % Extract training region
            training_window = rd_mag(range_idx-training_cells-guard_cells:range_idx+training_cells+guard_cells, ...
                                   doppler_idx-training_cells-guard_cells:doppler_idx+training_cells+guard_cells);
            
            guard_window = rd_mag(range_idx-guard_cells:range_idx+guard_cells, ...
                                doppler_idx-guard_cells:doppler_idx+guard_cells);
            training_cells_values = training_window(:);
            training_cells_values = setdiff(training_cells_values, guard_window(:));
            
            % Calculate threshold
            noise_level = mean(training_cells_values);
            threshold(range_idx, doppler_idx) = alpha * noise_level;

            if cut > threshold(range_idx, doppler_idx)
                detections(range_idx, doppler_idx) = true;
            end
        end
    end
end

